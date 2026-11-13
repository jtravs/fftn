# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Compiled, tested examples quoted by docs/ (AGENTS.md section 4). Each is a program that
# exits with 0 when its results are right, run as a test.

foreach(example IN ITEMS rows_c2c)
  add_executable(fftn_example_${example} ${example}.cpp)
  fftn_configure_target(fftn_example_${example})
  target_link_libraries(fftn_example_${example} PRIVATE fftn::fftn)
  add_test(NAME example.${example} COMMAND fftn_example_${example})
endforeach()
