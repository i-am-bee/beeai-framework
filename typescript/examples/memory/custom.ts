import { BaseMemory } from "beeai-framework/memory/base";
import { Message } from "beeai-framework/backend/message";
import { NotImplementedError } from "beeai-framework/errors";

export class MyMemory extends BaseMemory {
  get messages(): readonly Message[] {
    throw new NotImplementedError("Method not implemented.");
  }

  add(message: Message, index?: number): Promise<void> {
    throw new NotImplementedError("Method not implemented.");
  }

  delete(message: Message): Promise<boolean> {
    throw new NotImplementedError("Method not implemented.");
  }

  reset(): void {
    throw new NotImplementedError("Method not implemented.");
  }

  createSnapshot(): unknown {
    throw new NotImplementedError("Method not implemented.");
  }

  loadSnapshot(state: ReturnType<typeof this.createSnapshot>): void {
    throw new NotImplementedError("Method not implemented.");
  }
}
