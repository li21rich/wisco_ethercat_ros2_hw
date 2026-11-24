"""Every hardware stage ends at rest: when it finishes, fails, or is stopped with Ctrl-C."""
import pytest

from arm_stack import cli


class FakeClient:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def run(monkeypatch):
    calls = []
    client = FakeClient()
    monkeypatch.setattr(cli, 'make_client', lambda args, arm, calibration: client)
    monkeypatch.setattr(cli, 'go_to_rest', lambda arm, calibration, c: calls.append(c) or 0)

    def go(outcome, *extra):
        def stage(*a, **k):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        monkeypatch.setattr(cli, 'run_stage', stage)
        try:
            code = cli.main(['validate', '--no-log', *extra])
        except BaseException as error:      # noqa: B902 - the test inspects it
            code = error
        return code, calls, client
    return go


class Passed:
    passed = True


def test_ctrl_c_goes_to_rest_then_closes(run):
    code, calls, client = run(KeyboardInterrupt())
    assert code == 130 and len(calls) == 1 and client.closed


def test_finished_stage_goes_to_rest(run):
    code, calls, client = run(Passed())
    assert code == 0 and len(calls) == 1 and client.closed


def test_crash_still_goes_to_rest_and_reraises(run):
    code, calls, client = run(RuntimeError('boom'))
    assert isinstance(code, RuntimeError) and len(calls) == 1 and client.closed


def test_no_rest_holds_where_it_ends(run):
    code, calls, client = run(KeyboardInterrupt(), '--no-rest')
    assert code == 130 and calls == [] and client.closed
