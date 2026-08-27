export type BuildInfo = {
  mode: string;
  buildLabel: string;
  gitSha: string;
  previewHostname: string;
};

const env = import.meta.env;
const explicitPreview = String(env.VITE_PREVIEW_MODE ?? "").toLowerCase();

export const IS_PREVIEW_MODE =
  env.MODE === "preview" || explicitPreview === "1" || explicitPreview === "true";

export const API_BASE_URL = String(env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

const gitSha = String(env.VITE_GIT_SHA ?? "unknown");
const runtimeHostname =
  typeof window === "undefined" ? "unknown" : window.location.hostname || "unknown";

export const BUILD_INFO: BuildInfo = Object.freeze({
  mode: env.MODE,
  buildLabel: String(env.VITE_BUILD_LABEL ?? `${env.MODE}-${gitSha.slice(0, 8)}`),
  gitSha,
  previewHostname: String(env.VITE_PREVIEW_HOSTNAME ?? runtimeHostname),
});
