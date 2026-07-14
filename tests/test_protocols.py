import typing

from ksi.protocols import ForumMessageContent, PersistenceObserver
from ksi.tokens import TokenUsageDict


def test_on_forum_message_uses_typed_payloads():
    # Guards the #949 E3 boundary: the forum-message protocol payload must stay
    # TypedDict-typed (content_json -> ForumMessageContent, token_usage ->
    # TokenUsageDict), not regress to a bare ``dict``.
    hints = typing.get_type_hints(PersistenceObserver.on_forum_message)
    assert hints["content_json"] is ForumMessageContent
    assert hints["token_usage"] is TokenUsageDict


def test_forum_message_content_keys_are_optional():
    # total=False: every documented key is optional, so an error message that
    # carries only a subset still type-checks.
    assert ForumMessageContent.__total__ is False
    assert set(ForumMessageContent.__annotations__) == {"phase", "error", "error_type"}
