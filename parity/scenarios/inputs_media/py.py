"""Python side of `inputs_media`: a remote image, an inline base64 image, a remote PDF and an
inline PDF as user content parts, then a text question. Mirrors examples/basic/remote_image.py,
local_image.py, remote_pdf.py and local_file.py."""

import asyncio

from agents import Agent, Runner

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"
PDF_URL = "https://www.berkshirehathaway.com/letters/2024ltr.pdf"
B64 = "aGVsbG8="


async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": [{"type": "input_image", "detail": "auto", "image_url": IMAGE_URL}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "detail": "auto", "image_url": f"data:image/jpeg;base64,{B64}"}
                ],
            },
            {"role": "user", "content": [{"type": "input_file", "file_url": PDF_URL}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_data": f"data:application/pdf;base64,{B64}",
                        "filename": "letter.pdf",
                    }
                ],
            },
            {"role": "user", "content": "What do you see in these?"},
        ],
    )
    print(result.final_output)


asyncio.run(main())
