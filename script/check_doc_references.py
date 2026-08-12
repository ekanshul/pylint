# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/pylint-dev/pylint/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/pylint/blob/main/CONTRIBUTORS.txt

"""Add ``:ref:`` roles around pylint's own names in the documentation.

Used by pre-commit on the documentation and the changelog fragments, for the
two kinds of names that have a documentation page of their own:

* a bare message symbol (like ``no-member``) becomes ``:ref:`no-member```, so
  that it links to the message's page. Current and renamed messages come from
  pylint's message store, deleted messages come from ``DELETED_MESSAGES_IDS``;
* an option cited as ``--jobs`` or ``py-version`` becomes a link to that
  option's entry in the exhaustive option list.

Only names that still exist are linked, so a changelog entry about an option
that has since been removed keeps its literal text.

The two kinds are not spelled the same way on purpose. A message symbol is
only linked when it appears bare, because inside an inline literal it is
almost always part of a command line or of a configuration snippet the reader
is meant to copy. An option name is also linked inside an inline literal,
because there it is a citation rather than something to copy.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

from pylint.checkers import initialize as initialize_checkers
from pylint.extensions import initialize as initialize_extensions
from pylint.lint import PyLinter
from pylint.message._deleted_message_ids import DELETED_MESSAGES_IDS

# Words that look like message symbols: lowercase words joined by dashes
# (a few old symbols like 'raw_input-builtin' also contain underscores).
MESSAGE_PATTERN = re.compile(r"(?P<name>[a-z0-9_]+(?:-[a-z0-9_]+)+)")

# An option cited as an inline literal: ``--jobs`` or ``py-version``.
OPTION_LITERAL_PATTERN = re.compile(r"``(?P<dashes>--)?(?P<name>[a-z0-9][a-z0-9_-]*)``")

# An option cited as a bare command line flag: --jobs. A '=' after the name
# means a full command line ('--jobs=2'), which reads better left alone.
OPTION_FLAG_PATTERN = re.compile(
    r"(?<![\w`\-=/.])(?P<dashes>--)(?P<name>[a-z][a-z0-9_-]*)(?![\w`\-=])"
)

# An option cited in running text with no markup at all, quoted or not:
# "the notes-rgx option", "'ignore-signatures' is now honored". Only names
# holding a dash are looked for, so that ordinary words are never candidates.
OPTION_UNMARKED_PATTERN = re.compile(
    r"(?<![\w`<>/.=\-])(?P<name>[a-z][a-z0-9_]*(?:-[a-z0-9_]+)+)(?![\w`<>/=\-])"
)

# Inline markup spans in which a name must be left alone: inline literals,
# roles (including existing ``:ref:``), link labels and emphasis (rst does
# not render nested inline markup).
INLINE_SPAN_PATTERN = re.compile(
    r"``[^`]+``|`[^`]+`|\*\*[^\s*][^*]*\*\*|\*[^\s*][^*]*\*"
)

# Spans in which an option name must be left alone even when it is spelled as
# an inline literal: an existing role and hyperlinks.
ROLE_PATTERN = re.compile(r":[a-z:]+:`[^`]*`|`[^`]+`_")

DIRECTIVE_PATTERN = re.compile(r"\.\. ([a-zA-Z-]+)::")

# Directives whose content is regular text where references render fine.
# The content of every other directive (code blocks, tables, toctrees,
# comments, ...) is left untouched.
TEXT_DIRECTIVES = frozenset(
    {
        "admonition",
        "attention",
        "caution",
        "danger",
        "deprecated",
        "error",
        "hint",
        "important",
        "note",
        "seealso",
        "tip",
        "topic",
        "versionadded",
        "versionchanged",
        "warning",
    }
)

# A section title's underline. Adding a reference to a title would make it
# longer than its underline, so titles are left alone.
TITLE_MARKUP_PATTERN = re.compile(r"^([=\-~\"'`^*+#_:.])\1+$")

# Characters that disqualify a message symbol when directly before/after it
# (member of an option like '--disable=no-member', a path, a filename,
# an 'msgid:symbol' pair, ...).
BEFORE_GUARDS = frozenset("`-/\\._:=")
AFTER_GUARDS = frozenset("`/\\_")

# Option names that read as plain english more often than as an option. They
# are only linked when they are spelled as a command line flag ('--ignore'),
# so that ordinary words don't silently become links: in the documentation
# they name a regular expression group, a test attribute or an API parameter
# about as often as they name the option, and only a human can tell which.
AMBIGUOUS_OPTION_NAMES = frozenset(
    {"confidence", "disable", "enable", "evaluation", "ignore", "notes", "reports"}
)


class Names(NamedTuple):
    """The message symbols and the option names that have a doc page."""

    messages: frozenset[str]
    options: frozenset[str]


def get_all_names() -> Names:
    """Collect every name worth linking, from a fully loaded linter."""
    linter = PyLinter()
    initialize_checkers(linter)
    initialize_extensions(linter)
    symbols: set[str] = set()
    for message in linter.msgs_store.messages:
        symbols.add(message.symbol)
        symbols.update(old_symbol for _, old_symbol in message.old_names)
    for deleted_messages in DELETED_MESSAGES_IDS.values():
        for deleted_message in deleted_messages:
            symbols.add(deleted_message.symbol)
            symbols.update(old_symbol for _, old_symbol in deleted_message.old_names)
    return Names(
        messages=frozenset(symbols),
        options=frozenset(
            option_name
            for checker in linter.get_checkers()
            for option_name, _ in checker.options
        ),
    )


def _indentation(line: str) -> int:
    return len(line) - len(line.lstrip())


def _option_matches(line: str, names: Names) -> Iterator[tuple[re.Match[str], str]]:
    """Yield the option citations of a line, earliest first, without overlaps."""
    roles = [match.span() for match in ROLE_PATTERN.finditer(line)]
    marked_up = roles + [match.span() for match in INLINE_SPAN_PATTERN.finditer(line)]
    candidates = sorted(
        (
            *OPTION_LITERAL_PATTERN.finditer(line),
            *OPTION_FLAG_PATTERN.finditer(line),
            *OPTION_UNMARKED_PATTERN.finditer(line),
        ),
        key=lambda match: match.start(),
    )
    previous_end = 0
    for match in candidates:
        name = match.group("name")
        if name not in names.options:
            continue
        dashes = match.groupdict().get("dashes")
        if name in AMBIGUOUS_OPTION_NAMES and not dashes:
            continue
        start, end = match.span()
        if start < previous_end:
            continue
        spans = marked_up if match.re is OPTION_UNMARKED_PATTERN else roles
        if any(span_start <= start < span_end for span_start, span_end in spans):
            continue
        previous_end = end
        yield match, f":ref:`{dashes or ''}{name} <{name}-option>`"


def _message_matches(line: str, names: Names) -> Iterator[tuple[re.Match[str], str]]:
    """Yield the bare message symbols of a line, earliest first."""
    masked_spans = [match.span() for match in INLINE_SPAN_PATTERN.finditer(line)]
    for match in MESSAGE_PATTERN.finditer(line):
        name = match.group("name")
        start, end = match.span()
        if name not in names.messages:
            continue
        if any(span_start <= start < span_end for span_start, span_end in masked_spans):
            continue
        before = line[start - 1] if start else ""
        after = line[end] if end < len(line) else ""
        if before in BEFORE_GUARDS or after in AFTER_GUARDS:
            continue
        if after == "." and end + 1 < len(line) and line[end + 1].isalnum():
            # A filename like 'bad-name.py', not the end of a sentence
            continue
        yield match, f":ref:`{name}`"


def _add_references_to_line(line: str, names: Names) -> str:
    if line.count("`") % 2:
        # Unbalanced backticks: part of a multi-line inline span (a link or
        # a literal continued on the next line), don't touch anything
        return line
    replacements = sorted(
        (*_option_matches(line, names), *_message_matches(line, names)),
        key=lambda item: item[0].start(),
    )
    result: list[str] = []
    last_end = 0
    for match, reference in replacements:
        start, end = match.span()
        if start < last_end:
            continue
        result.append(line[last_end:start])
        result.append(reference)
        last_end = end
    result.append(line[last_end:])
    return "".join(result)


def add_references(content: str, names: Names) -> str:
    """Add ``:ref:`` roles to an rst document, leaving literal content alone."""
    lines = content.splitlines(keepends=True)
    new_lines: list[str] = []
    # While set, lines indented more than this are part of a literal block,
    # a non-text directive or a comment, and are left untouched.
    skipped_block_indentation: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if skipped_block_indentation is not None:
            if not stripped or _indentation(line) > skipped_block_indentation:
                new_lines.append(line)
                continue
            skipped_block_indentation = None
        if stripped.startswith("..") and (stripped == ".." or stripped[2] in " _|["):
            # Directive, comment, anchor, substitution or citation
            directive_match = DIRECTIVE_PATTERN.match(stripped)
            if (
                directive_match is None
                or directive_match.group(1) not in TEXT_DIRECTIVES
            ):
                skipped_block_indentation = _indentation(line)
            new_lines.append(line)
            continue
        if stripped.startswith(">>>"):
            # Doctest blocks
            new_lines.append(line)
            continue
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if TITLE_MARKUP_PATTERN.match(next_line):
            # A section title, whose underline would become too short
            new_lines.append(line)
            continue
        new_lines.append(_add_references_to_line(line, names))
        if stripped.endswith("::"):
            # The paragraph introduces a literal block
            skipped_block_indentation = _indentation(line)
    return "".join(new_lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="+", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Only report issues, do not modify files"
    )
    args = parser.parse_args(argv)
    names = get_all_names()
    return_value = 0
    for filename in args.filenames:
        content = filename.read_text(encoding="utf-8")
        new_content = add_references(content, names)
        if new_content == content:
            continue
        return_value = 1
        if args.check:
            print(f"{filename}: message symbols and option names should use ':ref:'")
        else:
            filename.write_text(new_content, encoding="utf-8")
            print(f"{filename}: added ':ref:' around pylint names")
    return return_value


if __name__ == "__main__":
    raise SystemExit(main())
