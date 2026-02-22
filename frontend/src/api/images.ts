import { API_BASE_URL } from "../config";
import type { Pose } from "../types/pose";

export type AllowedMoves = {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  turnLeft: boolean;
  turnRight: boolean;
};

type GetImagesResponse = {
  image_base64?: string;
  image?: string;
  imageBase64?: string;
  base64?: string;
  mimeType?: string;
  allowed?: unknown;
};

function getBase64Field(response: GetImagesResponse): string {
  return (
    response.image_base64 ??
    response.image ??
    response.imageBase64 ??
    response.base64 ??
    ""
  );
}

function parseAllowed(allowed?: unknown): AllowedMoves {
  const all = { forward: true, backward: true, left: true, right: true, turnLeft: true, turnRight: true };
  if (!allowed) return all;

  // Array of direction strings: ["forward", "left", "turnLeft"]
  if (Array.isArray(allowed)) {
    return {
      forward: allowed.includes("forward"),
      backward: allowed.includes("backward"),
      left: allowed.includes("left"),
      right: allowed.includes("right"),
      turnLeft: allowed.includes("turnLeft"),
      turnRight: allowed.includes("turnRight"),
    };
  }

  // Object like { forward: true, backward: false, turnLeft: true }
  if (typeof allowed === "object") {
    const obj = allowed as Record<string, boolean>;
    return {
      forward: !!obj.forward,
      backward: !!obj.backward,
      left: !!obj.left,
      right: !!obj.right,
      turnLeft: !!obj.turnLeft,
      turnRight: !!obj.turnRight,
    };
  }

  return all;
}

export type FetchImageResult = {
  imageSrc: string;
  allowed: AllowedMoves;
};

export async function fetchImageForPose(pose: Pose): Promise<string>;
export async function fetchImageForPose(pose: Pose, opts: { withAllowed: true }): Promise<FetchImageResult>;
export async function fetchImageForPose(pose: Pose, opts?: { withAllowed: boolean }): Promise<string | FetchImageResult> {
  const params = new URLSearchParams({
    x: String(pose.x),
    y: String(pose.y),
    z: String(pose.z),
    yaw: String(pose.yaw),
  });

  const base = (API_BASE_URL && API_BASE_URL !== "") ? API_BASE_URL : (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
  const url = new URL("/getImage", base);
  url.search = params.toString();

  const res = await fetch(url.toString(), {
    method: "GET",
  });

  if (!res.ok) {
    throw new Error(`Request failed (${res.status})`);
  }

  const payload = (await res.json()) as GetImagesResponse;
  const base64 = getBase64Field(payload).trim();

  if (!base64) {
    throw new Error("Image payload is empty");
  }

  const mime = payload.mimeType ?? "image/jpeg";
  const imageSrc = `data:${mime};base64,${base64}`;

  if (opts?.withAllowed) {
    console.log("API allowed field:", payload.allowed);
    return { imageSrc, allowed: parseAllowed(payload.allowed) };
  }
  return imageSrc;
}

