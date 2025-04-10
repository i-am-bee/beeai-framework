/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { MilvusDatabaseTool, MilvusToolOptions, MilvusAction } from "@/tools/database/milvus.js";

const mockClient = {
  listCollections: vi.fn(),
  describeCollection: vi.fn(),
  insert: vi.fn(),
  search: vi.fn(),
  delete: vi.fn(),
};

vi.mock("@zilliz/milvus2-sdk-node", () => ({
  MilvusClient: vi.fn(() => mockClient),
}));

describe("MilvusDatabaseTool", () => {
  let milvusDatabaseTool: MilvusDatabaseTool;

  beforeEach(() => {
    vi.clearAllMocks();
    milvusDatabaseTool = new MilvusDatabaseTool({
      connection: { address: "localhost:19530" },
    } as MilvusToolOptions);
  });

  it("throws a missing collection name error", async () => {
    await expect(
      milvusDatabaseTool.run({ action: MilvusAction.GetCollectionInfo }),
    ).rejects.toThrow(
      "Collection name is required for GetCollectionInfo, Search, Insert, and Delete actions.",
    );
  });

  it("throws missing collection name and vector error", async () => {
    await expect(milvusDatabaseTool.run({ action: MilvusAction.Search })).rejects.toThrow(
      "Both collection name and vector are required for Search action.",
    );
  });

  it("should get appropriate collection info", async () => {
    const collectionName = "test_collection";
    const mockCollectionInfo = {
      schema: { fields: [{ name: "vector", type: "FLOAT_VECTOR", params: { dim: 128 } }] },
    };
    mockClient.describeCollection.mockResolvedValueOnce(mockCollectionInfo);

    const response = await milvusDatabaseTool.run({
      action: MilvusAction.GetCollectionInfo,
      collectionName,
    });
    expect(response.result).toEqual(mockCollectionInfo);
  });

  it("performs a search on the collection", async () => {
    const collectionName = "dummy_collection";
    const vector = [0.1, 0.2, 0.3];
    const mockSearchResponse = { results: [{ id: 123, distance: 0.5 }] };
    mockClient.search.mockResolvedValueOnce(mockSearchResponse);

    const response = await milvusDatabaseTool.run({
      action: MilvusAction.Search,
      collectionName,
      vector,
      topK: 1,
    });
    expect(response.result).toEqual([{ id: 123, distance: 0.5 }]);
  });

  it("should delete from a collection correctly", async () => {
    const collectionName = "foobar_collection";
    const ids = [1, 2, 3];

    mockClient.delete.mockResolvedValueOnce({ deletedCount: 3 });

    const response = await milvusDatabaseTool.run({
      action: MilvusAction.Delete,
      collectionName,
      ids,
    });

    expect(mockClient.delete).toHaveBeenCalledWith({
      collection_name: collectionName,
      filter: `id in [${ids.join(",")}]`,
    });

    expect(response.result).toEqual({ deletedCount: 3 });
  });

  it("should handle empty collection list", async () => {
    const mockCollections = { data: [] };

    mockClient.listCollections.mockResolvedValueOnce(mockCollections);

    const response = await milvusDatabaseTool.run({ action: MilvusAction.ListCollections });

    expect(mockClient.listCollections).toHaveBeenCalledWith({});
    expect(response.result).toEqual([]);
  });
});
