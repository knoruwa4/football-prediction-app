import { proxyBackend } from "@/lib/backend";

export function POST(request: Request) {
  return proxyBackend(request, "api/predict");
}
