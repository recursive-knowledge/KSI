"""The forum package re-exports the public prompt-builder API."""


def test_forum_package_reexports_public_api():
    from kcsi.forum import (
        ForumPromptParts,
        build_cross_task_discussion_parts,
        build_per_task_discussion_parts,
    )

    assert ForumPromptParts is not None
    assert callable(build_per_task_discussion_parts)
    assert callable(build_cross_task_discussion_parts)


def test_private_inject_chars_accessible_via_prompt_module():
    from kcsi.forum.prompt import _NATIVE_MEMORY_FORUM_INJECT_CHARS

    assert isinstance(_NATIVE_MEMORY_FORUM_INJECT_CHARS, int)
