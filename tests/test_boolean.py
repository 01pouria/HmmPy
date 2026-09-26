def test_boolean_true(make_session):
    hmm = make_session([
        '{"value": true, "yes_probability": 0.9, "explanation": "supported"}',
        '{"value": true, "yes_probability": 0.8, "explanation": "supported"}',
        '{"value": true, "yes_probability": 0.85, "explanation": "supported"}',
    ])
    result = hmm.boolean("test", samples=3)
    assert result.value is True
    assert round(result.yes_probability, 4) == 0.85
    assert result.usage.requests == 3
    assert result.failed_checks == {}

def test_boolean_abstains(make_session):
    hmm = make_session([
        '{"value": true, "yes_probability": 0.60}',
        '{"value": true, "yes_probability": 0.60}',
        '{"value": true, "yes_probability": 0.60}',
    ])
    result = hmm.boolean("test", samples=3, min_probability=0.90, min_confidence=0.80)
    assert hmm.is_abstained(result)
    assert "probability" in result.failed_checks
    assert "confidence" in result.failed_checks
