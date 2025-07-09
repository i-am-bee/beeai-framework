/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
