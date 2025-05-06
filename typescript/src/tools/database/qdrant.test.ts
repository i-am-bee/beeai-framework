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
import { QdrantDatabaseTool, QdrantToolOptions, QdrantAction } from "@/tools/database/qdrant.js";

const mockClient = {
  getCollections: vi.fn(),
  getCollection: vi.fn(),
  upsert: vi.fn(),
  query: vi.fn(),
  delete: vi.fn(),
};

vi.mock("@qdrant/js-client-rest", () => ({
  QdrantClient: vi.fn(() => mockClient),
}));

describe("QdrantDatabaseTool", () => {
  let qdrantDatabaseTool: QdrantDatabaseTool;

  beforeEach(() => {
    vi.clearAllMocks();
    qdrantDatabaseTool = new QdrantDatabaseTool({
      connection: { url: "http://localhost:6333" },
    } as QdrantToolOptions);
  });

  it("throws a missing collection name error", async () => {
    await expect(
      qdrantDatabaseTool.run({ action: QdrantAction.GetCollectionInfo }),
    ).rejects.toThrow(
      "Collection name is required for GetCollectionInfo, Search, Insert, and Delete actions.",
    );
  });

  it("throws missing collection name and vector error", async () => {
    await expect(
      qdrantDatabaseTool.run({ action: QdrantAction.Search, collectionName: "test" }),
    ).rejects.toThrow("Vector is required for Search action.");
  });

  it("should get appropriate collection info", async () => {
    const collectionName = "test_collection";
    const mockCollectionInfo = {
      usage: {
        cpu: 1,
        payload_io_read: 1,
        payload_io_write: 1,
      },
      time: 0.002,
      status: "ok",
      result: {
        status: "green",
        optimizer_status: "ok",
        segments_count: 1,
        config: {
          params: {},
          hnsw_config: {
            m: 1,
            ef_construct: 1,
            full_scan_threshold: 1,
          },
        },
      },
    };

    mockClient.getCollection.mockResolvedValueOnce(mockCollectionInfo);

    const response = await qdrantDatabaseTool.run({
      action: QdrantAction.GetCollectionInfo,
      collectionName,
    });

    expect(mockClient.getCollection).toHaveBeenCalledWith(collectionName);
    expect(response.result).toEqual(mockCollectionInfo);
  });

  it("performs a search on the collection", async () => {
    const collectionName = "dummy_collection";
    const vector = [0.1, 0.2, 0.3];
    const mockSearchResponse = [{ id: "123", score: 0.95, payload: { name: "test document" } }];

    mockClient.query.mockResolvedValueOnce(mockSearchResponse);

    const response = await qdrantDatabaseTool.run({
      action: QdrantAction.Search,
      collectionName,
      vector,
      topK: 1,
    });

    expect(mockClient.query).toHaveBeenCalledWith(collectionName, {
      query: vector,
      limit: 1,
      with_payload: true,
    });
    expect(response.result).toEqual(mockSearchResponse);
  });

  it("should insert into a collection correctly", async () => {
    const collectionName = "embeddings_collection";
    const ids = [32, 532];
    const vectors = [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ];
    const payload = [{ name: "doc1" }, { name: "doc2" }];
    const mockInsertResponse = { operation_id: 123, status: "acknowledged" };

    mockClient.upsert.mockResolvedValueOnce(mockInsertResponse);

    const response = await qdrantDatabaseTool.run({
      action: QdrantAction.Insert,
      collectionName,
      vectors,
      payload,
      ids,
    });

    expect(mockClient.upsert).toHaveBeenCalledWith(collectionName, {
      points: [
        { id: 32, vector: vectors[0], payload: payload[0] },
        { id: 532, vector: vectors[1], payload: payload[1] },
      ],
    });
    expect(response.result).toEqual(mockInsertResponse);
  });

  it("should delete from a collection correctly", async () => {
    const collectionName = "foobar_collection";
    const ids = [1, 2, 3];
    const mockDeleteResponse = { operation_id: 456, status: "acknowledged" };

    mockClient.delete.mockResolvedValueOnce(mockDeleteResponse);

    const response = await qdrantDatabaseTool.run({
      action: QdrantAction.Delete,
      collectionName,
      ids,
    });

    expect(mockClient.delete).toHaveBeenCalledWith(collectionName, {
      points: [1, 2, 3],
    });
    expect(response.result).toEqual(mockDeleteResponse);
  });

  it("should handle empty collection list", async () => {
    const mockCollections = { collections: [] };

    mockClient.getCollections.mockResolvedValueOnce(mockCollections);

    const response = await qdrantDatabaseTool.run({ action: QdrantAction.ListCollections });

    expect(mockClient.getCollections).toHaveBeenCalled();
    expect(response.result).toEqual([]);
  });

  it("should list all collections", async () => {
    const mockCollections = {
      collections: [{ name: "collection1" }, { name: "collection2" }],
    };

    mockClient.getCollections.mockResolvedValueOnce(mockCollections);

    const response = await qdrantDatabaseTool.run({ action: QdrantAction.ListCollections });

    expect(mockClient.getCollections).toHaveBeenCalled();
    expect(response.result).toEqual(["collection1", "collection2"]);
  });
});
