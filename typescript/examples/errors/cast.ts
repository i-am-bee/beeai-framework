import { FrameworkError } from "beeai-framework/errors";

function getUser() {
  throw new Error("User was not found!");
}

try {
  getUser();
} catch (e) {
  const err = FrameworkError.ensure(e);
  console.log(err.dump());
  console.log(err.explain());
}
