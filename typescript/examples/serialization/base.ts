import { Serializer } from "beeai-framework/serializer/serializer";

const original = new Date("2024-01-01T00:00:00.000Z");
const serialized = await Serializer.serialize(original);
const deserialized = await Serializer.deserialize(serialized);

console.info(deserialized instanceof Date); // true
console.info(original.toISOString() === deserialized.toISOString()); // true
