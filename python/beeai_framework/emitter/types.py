# Copyright 2025 IBM Corp.
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


from pydantic import BaseModel, ConfigDict


class EventTrace(BaseModel):
    id: str
    run_id: str
    parent_run_id: str | None = None


class EmitterOptions(BaseModel):
    is_blocking: bool | None = None
    once: bool | None = None
    persistent: bool | None = None
    match_nested: bool | None = None

    model_config = ConfigDict(frozen=True)
