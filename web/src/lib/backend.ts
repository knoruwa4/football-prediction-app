export async function proxyBackend(
  request: Request,
  path: string,
): Promise<Response> {
  const baseUrl = process.env.FOOTBALL_API_URL ?? "http://127.0.0.1:8000";
  const incoming = new URL(request.url);
  const target = new URL(path, `${baseUrl.replace(/\/$/, "")}/`);
  target.search = incoming.search;

  try {
    const upstream = await fetch(target, { cache: "no-store" });
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json(
      { error: "Could not connect to the football prediction API." },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
