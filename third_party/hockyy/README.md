# hockyy/tesseract-packing (vendored)

Source files from https://github.com/hockyy/tesseract-packing by hockyy, with credit to the author.
They're vendored so the polish step survives reboots, since the old copy lived in /tmp. `UPSTREAM` holds the commit they came from.
`compat/bits/stdc++.h` is our shim for Apple clang. It's not part of upstream.

Build (macOS; the Command Line Tools linker is broken here, so use Xcode's toolchain):

    ./third_party/hockyy/build.sh
