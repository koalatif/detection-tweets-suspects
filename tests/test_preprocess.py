"""Tests unitaires du prétraitement et cohérence des artefacts."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_text

def test_lowercase():
    assert clean_text("HELLO World") == clean_text("hello world")

def test_removes_urls():
    assert "http" not in clean_text("check this http://bit.ly/x now")
    assert "www" not in clean_text("visit www.example.com today")

def test_removes_mentions_and_hashtags():
    out = clean_text("@user love #python coding")
    assert "user" not in out.split()

def test_removes_special_chars_and_digits():
    out = clean_text("Wow!!! 1234 $$$ cool")
    assert all(c.isalpha() or c == " " for c in out)

def test_removes_stopwords():
    assert "the" not in clean_text("the cat and the dog").split()

def test_non_empty_on_content():
    assert len(clean_text("winning amazing prizes")) > 0

def test_empty_input():
    assert clean_text("!!! @@@ 123") == ""
