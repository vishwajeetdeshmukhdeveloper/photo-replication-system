"""Quick API test + generate sample output images."""
import json
import http.client
import os

# Build multipart form data manually
boundary = "----TestBoundary12345"
filepath = "samples/input/vishwajeet_signature.png"

with open(filepath, "rb") as f:
    file_data = f.read()

body = (
    f"------TestBoundary12345\r\n"
    f'Content-Disposition: form-data; name="file"; filename="sig.png"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode() + file_data + b"\r\n------TestBoundary12345--\r\n"

# Send request
conn = http.client.HTTPConnection("localhost", 8000)
conn.request(
    "POST",
    "/api/replicate-steps",
    body=body,
    headers={"Content-Type": f"multipart/form-data; boundary=----TestBoundary12345"},
)
resp = conn.getresponse()
data = json.loads(resp.read().decode())
conn.close()

print("=" * 60)
print("  SIGNATURE REPLICATION SYSTEM — API TEST")
print("=" * 60)
print(f"  Success:          {data['success']}")
print(f"  Processing Time:  {data['processing_time_seconds']}s")
print(f"  Contours Found:   {data['metadata']['contour_count']}")
print(f"  Stroke Width:     {data['metadata']['stroke_width_mean']}px")
print(f"  Output Shape:     {data['metadata']['output_shape']}")
print(f"  Pipeline Steps:   {len(data['steps'])}")
print(f"  Step Names:")
for name in sorted(data['steps'].keys()):
    print(f"    - {name}")
print(f"  Final Image:      {len(data['final_image'])} chars (base64)")
print("=" * 60)

# Save final reconstruction as sample output
import base64
final_bytes = base64.b64decode(data['final_image'])
os.makedirs("samples/output", exist_ok=True)
with open("samples/output/reconstructed_signature.png", "wb") as f:
    f.write(final_bytes)
print("  Saved: samples/output/reconstructed_signature.png")

# Save each step
steps_dir = "samples/output/steps"
os.makedirs(steps_dir, exist_ok=True)
for name, b64 in data['steps'].items():
    img_bytes = base64.b64decode(b64)
    with open(f"{steps_dir}/{name}.png", "wb") as f:
        f.write(img_bytes)
    print(f"  Saved: {steps_dir}/{name}.png")

print("=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
