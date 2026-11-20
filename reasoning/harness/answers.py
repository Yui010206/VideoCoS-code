"""Deterministic answer formatting using only the public question and options.

This module never reads labels, scores, or scene evidence. It neither chooses an
answer nor converts units: ambiguous formatting is returned to the model to fix.
"""
from decimal import Decimal, InvalidOperation
import math
import re
import string

PREFIX = re.compile(r'^(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*)', re.I)
OPTION = re.compile(r'^\s*(?:\(([A-Z])\)|([A-Z])[.):])\s*(.*)$', re.I | re.S)
LETTER = re.compile(r'(?:\(([A-Z])\)|([A-Z])[.)]?)', re.I)
NUMBER = re.compile(r'^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
APPROXIMATE = re.compile(r'^(?:~|≈|about|approximately|approx\.?|around|roughly)\s*', re.I)
WORDS = {word: str(value) for value, word in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
    'fifteen sixteen seventeen eighteen nineteen twenty'.split())}
UNITS = {
    'm': ('m', 'meter', 'meters', 'metre', 'metres'),
    'cm': ('cm', 'centimeter', 'centimeters', 'centimetre', 'centimetres'),
    'mm': ('mm', 'millimeter', 'millimeters', 'millimetre', 'millimetres'),
    'm2': ('m2', 'm²', 'm^2', 'sq m', 'sq. m', 'sqm', 'square meter', 'square meters',
           'square metre', 'square metres'),
    'ft': ('ft', 'foot', 'feet'), 'in': ('inch', 'inches'),
    's': ('s', 'sec', 'second', 'seconds'),
}


def _options(task):
    options = task.get('options') or []
    if not isinstance(options, list) or any(not isinstance(o, str) for o in options):
        raise ValueError('Public options must be a list of strings')
    if len(options) > len(string.ascii_uppercase):
        raise ValueError('Too many public options for single-letter answers')
    result = {}
    for index, option in enumerate(options):
        match = OPTION.fullmatch(option)
        letter, text = ((match[1] or match[2]).upper(), match[3]) if match else (string.ascii_uppercase[index], option)
        if letter in result:
            raise ValueError('Public options have duplicate letters')
        result[letter] = text.strip()
    return result


def answer_format(task):
    options = _options(task)
    if options:
        return {'kind': 'choice', 'allowed_letters': list(options),
                'instruction': 'Submit one available option letter in text. Put explanation in explanation.'}
    return {'kind': 'number',
            'instruction': 'Submit one finite number in the question\'s requested units, without a unit suffix. Put explanation in explanation.'}


def _unit(text):
    return next((unit for unit, aliases in UNITS.items() if text.casefold() in aliases), None)


def _question_units(question):
    found = set()
    # Match longer area names first so "square meters" is not also a length.
    value = question.casefold()
    for unit in sorted(UNITS, key=lambda key: -max(map(len, UNITS[key]))):
        for alias in sorted(UNITS[unit], key=len, reverse=True):
            pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
            if re.search(pattern, value):
                found.add(unit)
                value = re.sub(pattern, ' ', value)
    return found


def _choice(text, options):
    """One letter, one exact option text, or a letter followed by its own option text."""
    matches = {letter for letter, option in options.items() if text.casefold() == option.casefold()}
    letter = LETTER.fullmatch(text)
    labelled = OPTION.fullmatch(text)
    if letter:
        matches.add((letter[1] or letter[2]).upper())
    elif labelled and options.get((labelled[1] or labelled[2]).upper(), '').casefold() == labelled[3].strip().casefold():
        matches.add((labelled[1] or labelled[2]).upper())
    if len(matches) == 1 and matches <= set(options):
        return matches.pop()
    raise ValueError('Ambiguous or unavailable answer. Submit exactly one of: ' + ', '.join(options))


def _number(text, question):
    text = APPROXIMATE.sub('', text, count=1)
    word = re.match(r'[a-z]+', text, re.I)
    if word and word[0].casefold() in WORDS:
        text = WORDS[word[0].casefold()] + text[word.end():]
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1].strip()
    match = NUMBER.match(text)
    if not match:
        raise ValueError('Submit one finite number, without prose, ranges, or alternative answers')
    suffix = text[match.end():].strip()
    if suffix:
        unit, wanted = _unit(suffix), _question_units(question)
        # Bare meters on an area question is a sloppy label, not a conversion.
        if unit is None or (wanted != {unit} and (unit, wanted) != ('m', {'m2'})):
            raise ValueError('Remove the unit suffix and submit one number in the question\'s requested units; no automatic unit conversion is performed')
    try:
        value = Decimal(match[0].replace(',', ''))
        approximate = float(value)
        if not value.is_finite() or not math.isfinite(approximate) or (approximate == 0 and value != 0):
            raise ValueError('Answer must be a finite number')
    except (InvalidOperation, OverflowError):
        raise ValueError('Answer must be a finite number') from None
    if value == 0:
        return '0'
    canonical = format(value, 'f')
    if '.' in canonical:
        canonical = canonical.rstrip('0').rstrip('.')
    return canonical if len(canonical) <= 200 else str(value).lower()


def normalize_answer(raw_text, task):
    """Return a canonical letter/number, or raise actionable format feedback.

    Accepted extras: an "Answer:" prefix, one trailing period, "A. <its option
    text>", a unit suffix equal to the question's unit, "~3"/"about 3", and
    number words up to twenty. Anything else is returned to the model to fix.
    """
    if not isinstance(raw_text, str) or not raw_text.strip() or len(raw_text) > 200:
        raise ValueError('Answer text must be a nonempty string of at most 200 characters')
    text = PREFIX.sub('', raw_text.strip(), count=1).strip()
    text = text[:-1].rstrip() if text.endswith('.') and len(text) > 1 else text
    options = _options(task)
    return _choice(text, options) if options else _number(text, str(task.get('question', '')))
