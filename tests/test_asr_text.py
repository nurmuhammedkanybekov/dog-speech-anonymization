"""Tests for the text scoring in vad_anonymization.asr (no model needed)."""

from __future__ import annotations

import math

from vad_anonymization.asr import normalize_text, word_error_rate, words_recovered


def test_normalize_drops_sound_tags_and_punctuation():
    assert normalize_text("(dog barks) Hello, World! [BLANK_AUDIO]") == "hello world"
    assert normalize_text("flour-fat") == "flour fat"


def test_normalize_drops_tags_cut_off_at_the_end():
    assert normalize_text("(wind blowing") == ""
    assert normalize_text("Come here (dog bar") == "come here"
    assert normalize_text("[door open") == ""


def test_wer():
    assert word_error_rate("a b c d", "a b c d") == 0.0
    assert word_error_rate("a b c d", "a x c") == 0.5          # 1 substitution + 1 deletion
    assert word_error_rate("a b", "(whispering)") == 1.0
    assert math.isnan(word_error_rate("", "a"))


def test_words_recovered_counts_each_word_once():
    assert words_recovered("the dog and the cat", "the cat") == 0.4
    assert words_recovered("a b", "") == 0.0
