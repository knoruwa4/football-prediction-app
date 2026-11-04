import { proxyBackend } from "@/lib/backend";

export function GET(request: Request) {
  return proxyBackend(request, "api/performance");
}
