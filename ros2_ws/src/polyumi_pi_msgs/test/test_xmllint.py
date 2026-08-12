# Copyright 2015 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Validate the package's XML against the ROS schema.

The only ament linter this package keeps: ruff covers the Python style rules the others
enforced, but nothing else checks package.xml. See package.xml for why the rest were dropped.
"""

import pytest
from ament_xmllint.main import main


@pytest.mark.linter
@pytest.mark.xmllint
def test_xmllint() -> None:
    """package.xml must parse and validate against package_format3.xsd."""
    rc = main(argv=[])
    assert rc == 0, 'Found code style errors / warnings'
