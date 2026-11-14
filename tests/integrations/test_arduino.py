from brickcell.integrations.arduino import ButtonGate


def test_uno_q_request_requires_fresh_readiness_and_new_press():
    context = dict(
        camera_ready=True,
        object_present=True,
        operator_confirmed=True,
        object_id="brick-1",
        session_id="session-1",
        camera_id="top",
        updated_at=100,
    )
    gate = ButtonGate()
    assert gate.poll(8, context, now=101) is None
    event = gate.poll(9, context, now=101)
    assert event["type"] == "capture.requested"
    assert event["robot_authorized"] is False
    assert event["decision"] is None
    assert gate.poll(9, context, now=101) is None
    assert gate.poll(10, context, now=110) is None
    context["operator_confirmed"] = False
    assert gate.poll(11, context, now=101) is None
    assert gate.poll(0, context, now=101) is None
