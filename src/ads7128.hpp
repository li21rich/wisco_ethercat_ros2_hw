// Copyright 2026 WiscoHumanoids
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ADS7128_HPP_
#define ADS7128_HPP_

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

namespace wisco
{

/// Minimal userspace driver for the TI ADS7128 (8-channel, 12-bit, I2C ADC).
///
/// Talks to the chip through the Linux i2c-dev interface (/dev/i2c-N) and uses
/// the ADC's manual mode: every read_raw() selects one channel (MANUAL_CHID),
/// then a plain 2-byte read starts a conversion and returns the result. The
/// chip stretches SCL while it converts.
///
/// Hardware oversampling: OSR_CFG can be set to average 2..128 conversions per
/// read. With averaging on, the chip stretches SCL until all samples are done
/// and returns a 16-bit result (12 integer bits + 4 fractional bits) instead of
/// the 12-bit code. read_raw() always reports counts in 12-bit units
/// (0..4095.9375) so callers do not have to care which mode is active.
/// No CRC, no software filtering.
class Ads7128
{
public:
  static constexpr int kNumChannels = 8;
  static constexpr uint16_t kFullScaleCounts = 4096;  // 12-bit converter

  Ads7128() = default;
  ~Ads7128() {close_device();}
  Ads7128(const Ads7128 &) = delete;
  Ads7128 & operator=(const Ads7128 &) = delete;

  /// Convert an oversampling ratio (1, 2, 4, ... 128) to the OSR_CFG[2:0] code.
  /// Returns false if the ratio is not a supported power of two.
  static bool osr_code_for_ratio(int ratio, uint8_t & code)
  {
    for (uint8_t c = 0; c <= 7; ++c) {
      if ((1 << c) == ratio) {
        code = c;
        return true;
      }
    }
    return false;
  }

  /// Open the bus, bind the slave address, reset and offset-calibrate the ADC,
  /// then program OSR_CFG. oversampling_ratio is 1 (off), 2, 4, 8, 16, 32, 64 or 128.
  bool open(const std::string & device, uint8_t address, int oversampling_ratio = 1)
  {
    close_device();
    uint8_t osr_code = 0;
    if (!osr_code_for_ratio(oversampling_ratio, osr_code)) {
      return fail_msg("oversampling ratio must be 1, 2, 4, 8, 16, 32, 64 or 128");
    }
    fd_ = ::open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
      return fail_errno("open " + device);
    }
    if (::ioctl(fd_, I2C_SLAVE, static_cast<unsigned long>(address)) < 0) {
      fail_errno("bind I2C address (EBUSY = a kernel driver already owns it)");
      close_device();
      return false;
    }
    if (!initialise(osr_code)) {
      close_device();
      return false;
    }
    return true;
  }

  /// Run one (possibly averaged) conversion on AIN<channel>.
  /// counts is in 12-bit units: 0..4095 without averaging, 0..4095.9375 with it.
  bool read_raw(uint8_t channel, double & counts)
  {
    if (fd_ < 0 || channel >= kNumChannels) {
      return fail_msg("read_raw: device not open or channel out of range");
    }
    // MANUAL_CHID lives in CHANNEL_SEL[3:0]; ZCD_CHID[7:4] stays 0.
    if (!write_reg(kRegChannelSel, channel)) {
      return false;
    }
    // The read frame's address ACK starts the conversion(s). Data is MSB-first.
    // Without averaging: D11..D0 left-justified, low nibble 0 (APPEND_STATUS off).
    // With averaging:    full 16-bit result, low nibble is the fractional part.
    // Either way, the 16-bit word / 16 is the value in 12-bit counts.
    uint8_t rx[2] = {0, 0};
    const ssize_t n = ::read(fd_, rx, sizeof(rx));
    if (n < 0) {
      return fail_errno("conversion read");
    }
    if (n != static_cast<ssize_t>(sizeof(rx))) {
      return fail_msg("conversion read: short read");
    }
    const uint16_t word = static_cast<uint16_t>((rx[0] << 8) | rx[1]);
    counts = static_cast<double>(word) / 16.0;
    return true;
  }

  const std::string & last_error() const {return error_;}

private:
  // Command opcodes (datasheet Table 9).
  static constexpr uint8_t kOpRegRead = 0x10;
  static constexpr uint8_t kOpRegWrite = 0x08;

  // Registers.
  static constexpr uint8_t kRegSystemStatus = 0x00;
  static constexpr uint8_t kRegGeneralCfg = 0x01;
  static constexpr uint8_t kRegOsrCfg = 0x03;
  static constexpr uint8_t kRegChannelSel = 0x11;

  // SYSTEM_STATUS bits.
  static constexpr uint8_t kStatusRsvd = 0x80;         // always reads 1
  static constexpr uint8_t kStatusCrcErrFuse = 0x04;   // power-up config CRC failed

  // GENERAL_CFG bits.
  static constexpr uint8_t kGenCfgRst = 0x01;
  static constexpr uint8_t kGenCfgCal = 0x02;

  // OSR_CFG[2:0]: 0 = no averaging, n = 2^n samples (max 7 = 128).
  static constexpr uint8_t kOsrMask = 0x07;

  bool initialise(uint8_t osr_code)
  {
    uint8_t status = 0;
    if (!read_reg(kRegSystemStatus, status)) {
      return false;
    }
    if (!(status & kStatusRsvd)) {
      return fail_msg("SYSTEM_STATUS bit 7 reads 0: device at this address is not an ADS7128");
    }

    // Soft reset to a known state: all pins analog, manual mode, CRC off, no averaging.
    if (!write_reg(kRegGeneralCfg, kGenCfgRst)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // tRST max is 5 ms

    if (!read_reg(kRegSystemStatus, status)) {
      return false;
    }
    if (!(status & kStatusRsvd)) {
      return fail_msg("no valid SYSTEM_STATUS after reset");
    }
    if (status & kStatusCrcErrFuse) {
      return fail_msg("ADS7128 reports a power-up configuration CRC error (CRC_ERR_FUSE)");
    }

    // ADC offset calibration; the chip clears the CAL bit when it is done.
    if (!write_reg(kRegGeneralCfg, kGenCfgCal)) {
      return false;
    }
    bool cal_done = false;
    for (int i = 0; i < 100 && !cal_done; ++i) {
      uint8_t cfg = 0;
      if (!read_reg(kRegGeneralCfg, cfg)) {
        return false;
      }
      if (!(cfg & kGenCfgCal)) {
        cal_done = true;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    if (!cal_done) {
      return fail_msg("ADC offset calibration did not complete");
    }

    // Hardware averaging. Written after the reset (which clears OSR_CFG) and read
    // back so a bus glitch can't silently leave averaging off.
    const uint8_t osr = static_cast<uint8_t>(osr_code & kOsrMask);
    if (!write_reg(kRegOsrCfg, osr)) {
      return false;
    }
    uint8_t readback = 0;
    if (!read_reg(kRegOsrCfg, readback)) {
      return false;
    }
    if ((readback & kOsrMask) != osr) {
      return fail_msg("OSR_CFG readback mismatch");
    }
    return true;
  }

  bool write_reg(uint8_t reg, uint8_t value)
  {
    const uint8_t tx[3] = {kOpRegWrite, reg, value};
    const ssize_t n = ::write(fd_, tx, sizeof(tx));
    if (n < 0) {
      return fail_errno("register write");
    }
    if (n != static_cast<ssize_t>(sizeof(tx))) {
      return fail_msg("register write: short write");
    }
    return true;
  }

  bool read_reg(uint8_t reg, uint8_t & value)
  {
    const uint8_t tx[2] = {kOpRegRead, reg};
    ssize_t n = ::write(fd_, tx, sizeof(tx));
    if (n < 0) {
      return fail_errno("register read (address phase)");
    }
    if (n != static_cast<ssize_t>(sizeof(tx))) {
      return fail_msg("register read: short write");
    }
    n = ::read(fd_, &value, 1);
    if (n < 0) {
      return fail_errno("register read (data phase)");
    }
    if (n != 1) {
      return fail_msg("register read: short read");
    }
    return true;
  }

  void close_device()
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  bool fail_errno(const std::string & what)
  {
    error_ = what + ": " + std::strerror(errno);
    return false;
  }

  bool fail_msg(const std::string & what)
  {
    error_ = what;
    return false;
  }

  int fd_{-1};
  std::string error_;
};

}  // namespace wisco

#endif  // ADS7128_HPP_