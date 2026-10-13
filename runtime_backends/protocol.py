"""Shared text serialization and strict native decision response validation."""
from __future__ import annotations

import json
import math
from typing import Any


def serialize_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_questions(questions: Any) -> dict[str, dict]:
    if not isinstance(questions, dict) or not questions:
        raise ValueError('questions must be a nonempty object')
    for name, question in questions.items():
        if not isinstance(name, str) or not isinstance(question, dict):
            raise ValueError('question names and definitions must be strings and objects')
        if not isinstance(question.get('instructions'), str):
            raise ValueError(f'{name}: instructions must be text')
        kind, criteria = question.get('type'), question.get('criteria')
        if kind == 'noul' and criteria is None:
            valid = True
        elif kind == 'score':
            valid = isinstance(criteria, list) and bool(criteria) and all(isinstance(x, str) for x in criteria)
        elif kind in ('choice', 'noul'):
            valid = isinstance(criteria, dict) and bool(criteria) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in criteria.items())
            if kind == 'noul':
                valid = valid and set(criteria) == {'false', 'true'}
        else:
            valid = False
        if not valid:
            raise ValueError(f'{name}: invalid typed criteria')
    return {name: {key: value for key, value in question.items()
                   if key in ("type", "instructions", "criteria")}
            for name, question in questions.items()}


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('probabilities must be numeric')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('probabilities must be finite and between zero and one')
    return value


def _distribution(answer: dict, labels: set[str], probability_decimals: int | None = None) -> dict[str, float]:
    probs = answer.get('probabilities')
    if not isinstance(probs, dict) or set(probs) != labels:
        raise ValueError('response must contain a complete probability distribution')
    values = {key: _probability(value) for key, value in probs.items()}
    tolerance = 1e-5 if probability_decimals is None else len(values) * 0.5 * 10 ** (-probability_decimals) + 1e-7
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0, abs_tol=tolerance):
        raise ValueError('probability distribution must sum to one')
    return values


def validate_prediction(result: Any, questions: dict[str, dict], probability_decimals: int | None = None) -> dict:
    if not isinstance(result, dict) or not isinstance(result.get('answers'), dict):
        raise ValueError('backend must return an answers object')
    if set(result['answers']) != set(questions):
        raise ValueError('backend must answer exactly the requested questions')
    for name, question in questions.items():
        answer = result['answers'][name]
        if not isinstance(answer, dict):
            raise ValueError(f'{name}: answer must be an object')
        if question['type'] == 'noul':
            p_true = _probability(answer.get('noul'))
            if 'probabilities' in answer:
                probs = _distribution(answer, {'false', 'true'}, probability_decimals)
                if not math.isclose(probs['true'], p_true, abs_tol=1e-5):
                    raise ValueError(f'{name}: noul must be P(true)')
        else:
            criteria = question['criteria']
            labels = set(criteria) if question['type'] == 'choice' else {str(i) for i in range(len(criteria))}
            probs = _distribution(answer, labels, probability_decimals)
            if question['type'] == 'choice' and answer.get('choice') not in labels:
                raise ValueError(f'{name}: choice is outside criteria')
            if question['type'] == 'score':
                score = answer.get('score')
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= len(criteria) - 1:
                    raise ValueError(f'{name}: score must be a finite criterion index')
        if 'confidence' in answer:
            _probability(answer['confidence'])
    json.dumps(result, allow_nan=False)
    return result


def canonical_answer(question: dict, probabilities: Any, boolean_yes_index: int = 1) -> dict:
    """Convert a native ordered readout without changing its probability semantics."""
    values = [_probability(value) for value in probabilities]
    expected = 2 if question['type'] == 'noul' else len(question['criteria'])
    if len(values) != expected or not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5):
        raise ValueError('native readout must contain a complete normalized probability vector')
    if question['type'] == 'noul':
        if boolean_yes_index not in (0, 1):
            raise ValueError('boolean_yes_index must be zero or one')
        p_true = values[boolean_yes_index]
        return {'noul': p_true, 'probabilities': {'false': values[1 - boolean_yes_index], 'true': p_true},
                'confidence': max(values)}
    labels = list(question['criteria']) if question['type'] == 'choice' else [str(i) for i in range(expected)]
    answer = {'probabilities': dict(zip(labels, values)), 'confidence': max(values)}
    if question['type'] == 'choice':
        answer['choice'] = labels[max(range(expected), key=values.__getitem__)]
    else:
        answer['score'] = sum(i * value for i, value in enumerate(values))
    return answer
