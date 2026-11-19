"""Exact text edits executed inside the disposable checkout."""

import inspect
import json

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Replace one exact occurrence of old_text with new_text in a UTF-8 "
            "checkout file. Omit old_text to create a new file in an existing directory. "
            "Use a relative path. Include enough surrounding text to make the match unique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "new_text"],
            "additionalProperties": False,
        },
    },
}

MAX_FILE_BYTES = 1_048_576


class InvalidEdit(ValueError):
    """The model can correct the tool call without abandoning its work."""


def edit_payload(tool, instructions_limit):
    try:
        args = json.loads(tool["function"]["arguments"])
        if not isinstance(args, dict) or args.keys() - {"path", "old_text", "new_text"}:
            raise ValueError
        path, old, new = args["path"], args.get("old_text"), args["new_text"]
        if (
            not isinstance(path, str)
            or len(path.encode()) > 1024
            or "\\" in path
            or any(ord(char) < 32 for char in path)
            or any(
                part.casefold() in {"", ".", "..", ".git"} for part in path.split("/")
            )
            or (old is not None and (not isinstance(old, str) or not old))
            or not isinstance(new, str)
            or old == new
            or any(
                len(part.encode()) > min(instructions_limit, MAX_FILE_BYTES)
                or "\0" in part
                for part in (old, new)
                if part is not None
            )
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise InvalidEdit(
            "Invalid edit_file arguments. Supply path and new_text. Supply a nonempty "
            "exact old_text to update an existing file. Use a relative checkout file path "
            "and keep the snippets within the patch size limit."
        ) from None
    return json.dumps(args).encode()


def _worker_edit(max_file_bytes):
    import json
    import os
    import stat
    import sys
    from contextlib import ExitStack
    from uuid import uuid4

    args = json.load(sys.stdin)
    *parents, name = args["path"].split("/")
    old, new = args.get("old_text"), args["new_text"]
    temp_name = f".gha-edit-{uuid4().hex}"
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    mode = 0o644

    with ExitStack() as handles:
        parent = os.open(".", directory_flags)
        handles.callback(os.close, parent)
        try:
            for part in parents:
                parent = os.open(part, directory_flags, dir_fd=parent)
                handles.callback(os.close, parent)
            if old is not None:
                current = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
                )
                with os.fdopen(current, "rb") as source:
                    info = os.fstat(source.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError("Target must be a regular checkout file.")
                    raw = source.read(max_file_bytes + 1)
                if len(raw) > max_file_bytes or b"\0" in raw:
                    raise ValueError("Target must be a bounded UTF-8 text file.")
                contents = raw.decode()
                first = contents.find(old)
                if first < 0 or contents.find(old, first + 1) >= 0:
                    raise ValueError(
                        "old_text must occur exactly once. Read the file again and use a unique exact match."
                    )
                new = contents[:first] + new + contents[first + len(old) :]
                mode = stat.S_IMODE(info.st_mode) & 0o777

            data = new.encode()
            if len(data) > max_file_bytes:
                raise ValueError("Edited file exceeds the size limit.")
            temp = os.open(
                temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, dir_fd=parent
            )
            try:
                with os.fdopen(temp, "wb") as destination:
                    os.fchmod(destination.fileno(), mode)
                    destination.write(data)
                if old is None:
                    os.link(temp_name, name, src_dir_fd=parent, dst_dir_fd=parent)
                else:
                    os.replace(temp_name, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(temp_name, dir_fd=parent)
                except FileNotFoundError:
                    pass
        except UnicodeError:
            sys.exit("Target must be a UTF-8 text file.")
        except ValueError as error:
            sys.exit(str(error))
        except FileExistsError:
            sys.exit("The target already exists. Supply an exact old_text to edit it.")
        except OSError:
            sys.exit(
                "Could not edit the file. Use an existing checkout directory and a regular file without symlinks."
            )
    print("Created file." if old is None else "Updated file.")


# Execute trusted source with isolated imports in the same sandbox as run commands.
EDIT_COMMAND = (
    "python3",
    "-I",
    "-c",
    f"{inspect.getsource(_worker_edit)}\n_worker_edit({MAX_FILE_BYTES})",
)
