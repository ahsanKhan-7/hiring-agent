from datetime import date

from conftest import GOOD, JD, OFF, STUFFER

from hiring_agent import scoring

TODAY = date(2026, 11, 22)


def test_good_beats_stuffer_beats_off_topic():
    ranked = scoring.score_resumes(JD, [OFF, STUFFER, GOOD], TODAY)
    assert [c["name"] for c in ranked] == ["good", "stuffer", "off"]
    assert [c["rank"] for c in ranked] == [1, 2, 3]


def test_listed_only_skills_are_reported_and_get_half_credit():
    by = {c["name"]: c for c in scoring.score_resumes(JD, [GOOD, STUFFER], TODAY)}
    assert len(by["stuffer"]["listed_only_skills"]) == 6 and by["good"]["listed_only_skills"] == []
    assert by["stuffer"]["contributions"]["skills"] < by["good"]["contributions"]["skills"]


def test_contributions_sum_to_score():
    for c in scoring.score_resumes(JD, [GOOD, STUFFER, OFF], TODAY):
        assert abs(sum(c["contributions"].values()) - c["score"]) < 0.02


def test_name_and_contact_details_do_not_change_score():
    a = {"name": "x", "text": "John Smith\njohn@x.com\nExperience\nJan 2024 - Dec 2024 DS\nPython sql"}
    b = {"name": "x", "text": "Aisha Khan\naisha@y.org\nExperience\nJan 2024 - Dec 2024 DS\nPython sql"}
    assert scoring.score_resumes(JD, [a], TODAY)[0]["score"] == scoring.score_resumes(JD, [b], TODAY)[0]["score"]


def test_experience_merges_overlaps_and_ignores_education():
    text = "Experience\nJan 2022 - Dec 2022 A\nJun 2022 - Dec 2023 B\nEducation\n2010 - 2014 BS"
    assert scoring.experience_years(text, TODAY) == round(23 / 12, 2)


def test_no_jd_skills_drops_skills_signal_instead_of_zeroing():
    c = scoring.score_resumes("a generic role", [GOOD], TODAY)[0]
    assert "skills" not in c["contributions"]


def test_empty_input():
    assert scoring.score_resumes(JD, []) == []
