import pytest
from main import get_user_input, validate_tv_shows, generate_recommendations

def test_get_user_input(monkeypatch):
    # Simulate user input
    monkeypatch.setattr('builtins.input', lambda _: "game of throns, lupan, witcher")
    result = get_user_input()
    assert result == ["game of throns", "lupan", "witcher"]

def test_validate_tv_shows():
    user_input = ["game of throns", "lupan", "witcher"]
    valid_shows = validate_tv_shows(user_input)
    assert valid_shows == ["Game Of Thrones", "Lupin", "The Witcher"]

def test_generate_recommendations():
    valid_shows = ["Game Of Thrones", "Lupin", "The Witcher"]
    recommendations = generate_recommendations(valid_shows)
    assert "Breaking Bad" in recommendations
    assert isinstance(recommendations["Breaking Bad"], float)
