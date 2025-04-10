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

import { ValueError } from "@/errors.js";

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new ValueError("Vectors must have equal length");
  }
  // (cos θ) = (A · B) / (|A| * |B|)
  const dot = vecA.reduce((sum, value, index) => sum + value * vecB[index], 0);
  const magA = Math.sqrt(vecA.reduce((sum, value) => sum + value ** 2, 0));
  const magB = Math.sqrt(vecB.reduce((sum, value) => sum + value ** 2, 0));

  if (magA === 0 || magB === 0) {
    throw new ValueError("Vectors cannot not have zero magnitude");
  }
  return dot / (magA * magB);
}

export function cosineSimilarityMatrix(matrixA: number[][], matrixB: number[][]): number[][] {
  if ((matrixA[0]?.length ?? 0) !== (matrixA[0]?.length ?? 0)) {
    throw new ValueError("Matrices must have the same number of columns.");
  }

  return matrixA.map((rowA) => matrixB.map((rowB) => cosineSimilarity(rowA, rowB)));
}
