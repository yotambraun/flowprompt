# Multimodal Guide

**Native support for images, documents, audio, and video**

FlowPrompt provides first-class support for multimodal inputs, enabling you to build prompts that work with images, documents, audio, and video content alongside text.

## Table of Contents

- [Quick Start](#quick-start)
- [Image Support](#image-support)
- [Document Support](#document-support)
- [Audio Support](#audio-support)
- [Video Support](#video-support)
- [Provider Compatibility](#provider-compatibility)
- [Best Practices](#best-practices)

## Quick Start

Here's a simple example of analyzing an image:

```python
from flowprompt.multimodal import VisionPrompt, ImageContent
from pydantic import BaseModel


# Define a vision prompt
class ImageAnalyzer(VisionPrompt):
    system = "You are an expert at analyzing images."
    user = "Describe what you see in this image."

    class Output(BaseModel):
        description: str
        objects: list[str]
        mood: str


# Analyze an image
result = ImageAnalyzer().with_image("photo.jpg").run(model="gpt-4o")

print(result.description)
print(f"Objects: {', '.join(result.objects)}")
print(f"Mood: {result.mood}")
```

## Image Support

### Basic Image Analysis

```python
from flowprompt.multimodal import VisionPrompt, ImageContent

# From file path
result = VisionPrompt.describe("photo.jpg").run(model="gpt-4o")

# From URL
result = VisionPrompt.describe("https://example.com/image.jpg").run(model="gpt-4o")

# With detail level
result = VisionPrompt.describe(
    "photo.jpg",
    detail_level="comprehensive",  # "brief", "comprehensive", "technical"
).run(model="gpt-4o")
```

### Creating ImageContent

Multiple ways to create image content:

```python
from flowprompt.multimodal import ImageContent

# From file path
image = ImageContent.from_file("photo.jpg", detail="high")

# From URL
image = ImageContent.from_url("https://example.com/image.jpg")

# From raw bytes
with open("photo.jpg", "rb") as f:
    data = f.read()
image = ImageContent.from_bytes(data, format=ImageFormat.JPEG)

# With metadata
image = ImageContent.from_file(
    "photo.jpg", detail="high", alt_text="Product photo showing red sneakers"
)
```

### Custom Vision Prompts

Build custom prompts for specific vision tasks:

```python
from flowprompt.multimodal import VisionPrompt, ImageContent
from pydantic import BaseModel


class ProductAnalyzer(VisionPrompt):
    """Analyze product images for e-commerce."""

    system = """You are an expert at analyzing product images.
    Extract key details about the product shown."""

    user = "Analyze this product image and extract details."

    class Output(BaseModel):
        product_type: str
        color: str
        condition: str
        estimated_value: float
        description: str


# Use with image
analyzer = ProductAnalyzer().with_image("product.jpg")
result = analyzer.run(model="gpt-4o")

print(f"Product: {result.product_type}")
print(f"Color: {result.color}")
print(f"Value: ${result.estimated_value:.2f}")
```

### Multiple Images

Compare or analyze multiple images:

```python
from flowprompt.multimodal import VisionPrompt

# Compare two images
result = VisionPrompt.compare(
    images=["before.jpg", "after.jpg"],
    comparison_type="differences",  # "general", "differences", "similarities"
).run(model="gpt-4o")


# Custom multi-image prompt
class MultiImageAnalyzer(VisionPrompt):
    system = "Compare and contrast the provided images."
    user = "What are the key differences and similarities?"


result = (
    MultiImageAnalyzer()
    .with_images(["image1.jpg", "image2.jpg", "image3.jpg"])
    .run(model="gpt-4o")
)
```

### Image Detail Levels

Control processing detail for cost/quality tradeoff:

```python
# Low detail - faster, cheaper
image = ImageContent.from_file("photo.jpg", detail="low")

# Auto detail - balanced (default)
image = ImageContent.from_file("photo.jpg", detail="auto")

# High detail - slower, more accurate
image = ImageContent.from_file("photo.jpg", detail="high")
```

## Document Support

### PDF Processing

Extract text and analyze PDF documents:

```python
from flowprompt.multimodal import DocumentPrompt, DocumentContent

# Quick summarization
result = DocumentPrompt.summarize(
    "report.pdf",
    length="medium",  # "brief", "medium", "detailed"
).run(model="gpt-4o")

print(result)


# Custom document analysis
class ContractAnalyzer(DocumentPrompt):
    system = "You are a legal document expert."
    user = "Extract key terms and obligations from this contract."

    class Output(BaseModel):
        parties: list[str]
        key_terms: list[str]
        obligations: list[str]
        expiration_date: str | None


result = ContractAnalyzer().with_document("contract.pdf").run(model="gpt-4o")
```

### Document Content Types

```python
from flowprompt.multimodal import DocumentContent, DocumentFormat

# PDF documents
doc = DocumentContent.from_file("report.pdf")

# DOCX files
doc = DocumentContent.from_file("document.docx")

# Plain text
doc = DocumentContent.from_file("readme.txt")

# HTML (with BeautifulSoup if available)
doc = DocumentContent.from_file("webpage.html")

# Markdown
doc = DocumentContent.from_file("guide.md")

# From text directly
doc = DocumentContent.from_text(
    "This is the document content...", format=DocumentFormat.TXT
)
```

### Document with Images

Extract text and render pages as images:

```python
# Extract pages as images for visual analysis
doc = DocumentContent.from_file(
    "report.pdf",
    extract_images=True,  # Render pages as images
    max_pages=10,  # Limit pages processed
)

# Access extracted content
print(doc.text)  # Extracted text
print(doc.page_count)  # Number of pages
print(len(doc.pages))  # Number of page images


# Use in prompt
class VisualDocumentAnalyzer(DocumentPrompt):
    system = "Analyze documents including their visual layout."
    user = "Describe the document structure and key visual elements."


result = (
    VisualDocumentAnalyzer()
    .with_document("report.pdf", extract_images=True)
    .run(model="gpt-4o")
)
```

### Information Extraction

Extract specific information from documents:

```python
# Extract entities
result = DocumentPrompt.extract_info(
    "article.pdf",
    info_type="entities",  # "key_facts", "entities", "dates", "numbers"
).run(model="gpt-4o")


# Custom extraction
class DataExtractor(DocumentPrompt):
    system = "Extract structured data from documents."
    user = "Extract all numerical data and create a structured summary."

    class Output(BaseModel):
        total_revenue: float
        expenses: list[dict[str, float]]
        profit_margin: float
        key_metrics: dict[str, float]


result = DataExtractor().with_document("financial_report.pdf").run(model="gpt-4o")
```

## Audio Support

### Audio Transcription and Analysis

```python
from flowprompt.multimodal import MultimodalPrompt, AudioContent
from pydantic import BaseModel


class AudioAnalyzer(MultimodalPrompt):
    system = "Transcribe and analyze audio content."
    user = "Transcribe this audio and summarize key points."

    class Output(BaseModel):
        transcription: str
        summary: str
        speakers: int
        topics: list[str]


# From file
audio = AudioContent.from_file("recording.mp3")
result = AudioAnalyzer(audio=audio).run(model="gpt-4o")

# From URL
audio = AudioContent.from_url("https://example.com/audio.mp3")
result = AudioAnalyzer(audio=audio).run(model="gpt-4o")
```

### Supported Audio Formats

```python
from flowprompt.multimodal import AudioContent, AudioFormat

# Supported formats
formats = [
    AudioFormat.MP3,  # .mp3
    AudioFormat.WAV,  # .wav
    AudioFormat.OGG,  # .ogg
    AudioFormat.FLAC,  # .flac
    AudioFormat.M4A,  # .m4a
    AudioFormat.WEBM,  # .webm
]

# Format is auto-detected from file extension
audio = AudioContent.from_file("recording.mp3")  # Detected as MP3
```

### Audio with Transcription

Provide pre-computed transcription for better results:

```python
audio = AudioContent.from_file(
    "recording.mp3", transcription="This is the pre-computed transcription..."
)

# The transcription can be used alongside or instead of audio
```

## Video Support

### Video Frame Extraction

Videos are processed by extracting frames:

```python
from flowprompt.multimodal import MultimodalPrompt, VideoContent


class VideoAnalyzer(MultimodalPrompt):
    system = "Analyze video content frame by frame."
    user = "Describe what happens in this video."


# Extract frames from video
video = VideoContent.from_file(
    "demo.mp4",
    frame_interval=1.0,  # Extract frame every 1 second
    max_frames=10,  # Maximum 10 frames
)

result = VideoAnalyzer(video=video).run(model="gpt-4o")
```

### Manual Frame Extraction

Control frame extraction separately:

```python
from flowprompt.multimodal import VideoContent

# Create video content
video = VideoContent.from_file("demo.mp4")

# Extract frames manually
frames = video.extract_frames()  # Returns list[ImageContent]

# Each frame has metadata
for frame in frames:
    print(f"Frame at {frame.metadata['timestamp_seconds']:.1f}s")


# Use frames directly
class FrameAnalyzer(VisionPrompt):
    system = "Analyze video frames."
    user = "Describe changes across these frames."


result = FrameAnalyzer().with_images(frames).run(model="gpt-4o")
```

### Video Metadata

Access video information:

```python
video = VideoContent.from_file("demo.mp4")
video.extract_frames()

print(f"Duration: {video.duration_seconds:.1f}s")
print(f"Frames extracted: {len(video.frames)}")
print(f"File size: {video.get_size_bytes() / 1024 / 1024:.2f} MB")
```

## Mixed Multimodal Content

Combine multiple content types in a single prompt:

```python
from flowprompt.multimodal import MultimodalPrompt, ImageContent, DocumentContent


class ComprehensiveAnalyzer(MultimodalPrompt):
    system = "Analyze all provided content comprehensively."
    user = "Compare the document content with the images and provide insights."

    class Output(BaseModel):
        summary: str
        consistency_check: str
        recommendations: list[str]


# Add multiple content types
analyzer = ComprehensiveAnalyzer()
analyzer = analyzer.add_image("chart.jpg")
analyzer = analyzer.add_image("graph.jpg")
analyzer = analyzer.add_document("report.pdf")

result = analyzer.run(model="gpt-4o")
```

### Content Position

Control where multimodal content appears:

```python
class FlexiblePrompt(MultimodalPrompt):
    system = "System message"
    user = "Analyze the provided content."


# Content before text (default)
prompt = FlexiblePrompt(
    images=[ImageContent.from_file("photo.jpg")], content_position="before"
)

# Content after text
prompt = FlexiblePrompt(
    images=[ImageContent.from_file("photo.jpg")], content_position="after"
)

# Inline (mixed)
prompt = FlexiblePrompt(
    images=[ImageContent.from_file("photo.jpg")], content_position="inline"
)
```

## Provider Compatibility

### OpenAI Models

Full support for vision and multimodal capabilities:

```python
# GPT-4 Vision
result = VisionPrompt.describe("photo.jpg").run(model="gpt-4o")

# GPT-4 Turbo
result = VisionPrompt.describe("photo.jpg").run(model="gpt-4-turbo")
```

### Anthropic Models

Claude models with vision support:

```python
# Claude 3 Opus
result = VisionPrompt.describe("photo.jpg").run(
    model="anthropic/claude-3-opus-20240229"
)

# Claude 3 Sonnet
result = VisionPrompt.describe("photo.jpg").run(
    model="anthropic/claude-3-5-sonnet-20241022"
)
```

### Google Models

Gemini models with multimodal support:

```python
# Gemini Pro Vision
result = VisionPrompt.describe("photo.jpg").run(model="gemini/gemini-pro-vision")

# Gemini 2.0 Flash
result = VisionPrompt.describe("photo.jpg").run(model="gemini/gemini-2.0-flash-exp")
```

### Feature Support Matrix

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Images | Yes | Yes | Yes |
| PDFs (as images) | Yes | Yes | Yes |
| PDF text extraction | Client-side | Client-side | Client-side |
| Audio | Limited | No | Yes |
| Video (frames) | Yes | Yes | Yes |
| Multiple images | Yes | Yes | Yes |

**Note:** Document text extraction happens client-side, so all providers can work with extracted text.

## Best Practices

### 1. Choose the Right Detail Level

Balance cost and accuracy:

```python
# For OCR and detailed analysis
image = ImageContent.from_file("document.jpg", detail="high")

# For general object recognition
image = ImageContent.from_file("photo.jpg", detail="auto")

# For classification or simple tasks
image = ImageContent.from_file("thumbnail.jpg", detail="low")
```

### 2. Optimize Image Sizes

Resize images before processing to reduce costs:

```python
from PIL import Image
import io


def optimize_image(path: str, max_size: int = 1024) -> ImageContent:
    """Resize image while maintaining aspect ratio."""
    img = Image.open(path)

    # Calculate new size
    ratio = min(max_size / img.width, max_size / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))

    # Resize
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return ImageContent.from_bytes(buffer.read(), format=ImageFormat.JPEG)


# Use optimized image
image = optimize_image("large_photo.jpg")
result = VisionPrompt.describe(image).run(model="gpt-4o")
```

### 3. Limit PDF Pages

Process only necessary pages to control costs:

```python
# Process first 10 pages only
doc = DocumentContent.from_file("large_report.pdf", max_pages=10)

# Or extract specific sections first using PyPDF2/pypdf
import pypdf

reader = pypdf.PdfReader("large_report.pdf")
writer = pypdf.PdfWriter()

# Extract pages 5-10
for page_num in range(5, 11):
    writer.add_page(reader.pages[page_num])

# Write to temporary file
with open("section.pdf", "wb") as f:
    writer.write(f)

# Process section
doc = DocumentContent.from_file("section.pdf")
```

### 4. Batch Similar Operations

Process multiple images in one call:

```python
# Instead of multiple calls
results = []
for image_path in image_paths:
    result = VisionPrompt.describe(image_path).run(model="gpt-4o")
    results.append(result)


# Do this (one call with multiple images)
class BatchAnalyzer(VisionPrompt):
    system = "Analyze each image and provide results."
    user = "Analyze all provided images."

    class Output(BaseModel):
        results: list[dict[str, str]]


result = BatchAnalyzer().with_images(image_paths).run(model="gpt-4o")
```

### 5. Handle Errors Gracefully

Multimodal operations can fail in various ways:

```python
from pathlib import Path


def safe_analyze_image(path: str) -> dict:
    """Safely analyze an image with error handling."""
    try:
        # Check file exists
        if not Path(path).exists():
            return {"error": "File not found"}

        # Check file size (< 20MB)
        if Path(path).stat().st_size > 20 * 1024 * 1024:
            return {"error": "File too large"}

        # Try analysis
        result = VisionPrompt.describe(path).run(model="gpt-4o")
        return {"success": True, "result": result}

    except Exception as e:
        return {"error": str(e)}


# Use it
result = safe_analyze_image("photo.jpg")
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(result["result"])
```

### 6. Cache Extracted Content

Reuse extracted content to avoid repeated processing:

```python
# Extract once
doc = DocumentContent.from_file("report.pdf", extract_images=True)
doc.process()  # Processes immediately

# Reuse in multiple prompts
summarizer = DocumentPrompt.summarize(doc)
result1 = summarizer.run(model="gpt-4o")


class DetailedAnalyzer(DocumentPrompt):
    system = "Provide detailed analysis."
    user = "Analyze this document in detail."


result2 = DetailedAnalyzer().with_document(doc).run(model="gpt-4o")
# Document is not re-processed!
```

### 7. Use Appropriate Models

Choose models based on your needs:

```python
# For complex visual analysis - use GPT-4 Vision
result = ComplexAnalyzer().with_image("complex_scene.jpg").run(model="gpt-4o")

# For simple classification - use faster/cheaper models
result = (
    SimpleClassifier()
    .with_image("product.jpg")
    .run(
        model="gpt-4o-mini"  # Cheaper, faster
    )
)
```

### 8. Validate Multimodal Input

Ensure content is valid before processing:

```python
from flowprompt.multimodal import ImageContent


def validate_and_create_image(path: str) -> ImageContent | None:
    """Validate image before creating ImageContent."""
    from PIL import Image

    try:
        # Try to open with PIL
        img = Image.open(path)

        # Check format
        if img.format not in ["JPEG", "PNG", "GIF", "WEBP"]:
            print(f"Unsupported format: {img.format}")
            return None

        # Check size
        if img.width < 10 or img.height < 10:
            print("Image too small")
            return None

        if img.width > 4096 or img.height > 4096:
            print("Image too large")
            return None

        return ImageContent.from_file(path)

    except Exception as e:
        print(f"Invalid image: {e}")
        return None


# Use it
image = validate_and_create_image("photo.jpg")
if image:
    result = VisionPrompt.describe(image).run(model="gpt-4o")
```

## Advanced Usage

### Custom Content Processing

Process content before sending to the model:

```python
from flowprompt.multimodal import DocumentContent
import re


class PreprocessedDocumentContent(DocumentContent):
    """Custom document content with preprocessing."""

    def process(self):
        super().process()

        if self.text:
            # Clean up text
            self.text = re.sub(r"\s+", " ", self.text)  # Normalize whitespace
            self.text = re.sub(r"[^\w\s.,!?-]", "", self.text)  # Remove special chars

            # Extract key sections
            sections = self.text.split("\n\n")
            self.metadata["section_count"] = len(sections)


# Use custom content
doc = PreprocessedDocumentContent.from_file("report.pdf")
result = DocumentPrompt.summarize(doc).run(model="gpt-4o")
```

### Streaming with Multimodal

Stream responses for multimodal prompts:

```python
# Streaming works normally with multimodal content
prompt = VisionPrompt.describe("photo.jpg")

for chunk in prompt.stream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)

# Async streaming
async for chunk in prompt.astream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

### Multimodal with Optimization

Optimize multimodal prompts:

```python
from flowprompt.optimize import optimize, ExampleDataset, Example, ExactMatch

# Create examples with images
dataset = ExampleDataset(
    [
        Example(
            input={"image_path": "product1.jpg"},
            output={"category": "electronics", "confidence": 0.95},
        ),
        Example(
            input={"image_path": "product2.jpg"},
            output={"category": "clothing", "confidence": 0.90},
        ),
        # More examples...
    ]
)


# Optimize vision prompt
class ProductClassifier(VisionPrompt):
    system = "Classify products from images."
    user = "What category is this product?"

    class Output(BaseModel):
        category: str
        confidence: float


result = optimize(
    ProductClassifier, dataset=dataset, metric=ExactMatch(), strategy="fewshot"
)
```

## Next Steps

- Learn about [Optimization](optimization.md) to improve multimodal prompts
- Check the [API Reference](api.md) for detailed documentation
- See [Examples](https://github.com/yotambraun/flowprompt/tree/main/examples) for more multimodal patterns

## Dependencies

Some multimodal features require additional packages:

```bash
# For PDF processing
pip install pypdf  # or pymupdf

# For DOCX processing
pip install python-docx

# For HTML processing (optional)
pip install beautifulsoup4

# For video frame extraction
pip install opencv-python

# For image optimization
pip install pillow
```

Install all multimodal dependencies:

```bash
pip install flowprompt[multimodal]
```
