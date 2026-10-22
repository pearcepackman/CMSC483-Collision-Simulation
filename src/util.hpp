#pragma once

// Stub bodies call this so an unfinished function stops the program loudly
// instead of silently skipping physics. Prints "not implemented: <what>".
[[noreturn]] void not_implemented(const char *what);
