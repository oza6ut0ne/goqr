package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/kettek/apng"
	"github.com/liyue201/goqr"
	"github.com/skip2/go-qrcode"
)

// The size of the generated PNG image in pixels.
const pngSize = 512

// Padding used in grid generation between QR codes and around the border.
const gridPadding = 20

func createGridPNG(images []image.Image, outputFilename string) {
	if len(images) == 0 {
		log.Println("No images to create a grid from.")
		return
	}

	// All QR codes are the same size.
	qrWidth := images[0].Bounds().Dx()
	qrHeight := images[0].Bounds().Dy()

	// Calculate grid dimensions to be as square as possible.
	numImages := len(images)
	cols := int(math.Ceil(math.Sqrt(float64(numImages))))
	rows := int(math.Ceil(float64(numImages) / float64(cols)))

	// Calculate final image dimensions with padding.
	gridWidth := cols*qrWidth + (cols+1)*gridPadding
	gridHeight := rows*qrHeight + (rows+1)*gridPadding

	// Create a new RGBA image to act as the canvas.
	gridImage := image.NewRGBA(image.Rect(0, 0, gridWidth, gridHeight))
	// Fill canvas with a white background.
	draw.Draw(gridImage, gridImage.Bounds(), image.White, image.Point{}, draw.Src)

	// Draw each QR code onto the grid.
	for i, img := range images {
		row := i / cols
		col := i % cols
		// Calculate the top-left corner for this QR code.
		x := gridPadding + col*(qrWidth+gridPadding)
		y := gridPadding + row*(qrHeight+gridPadding)
		rect := image.Rect(x, y, x+qrWidth, y+qrHeight)
		draw.Draw(gridImage, rect, img, image.Point{}, draw.Src)
	}

	// Save the final grid image.
	outFile, err := os.Create(outputFilename)
	if err != nil {
		log.Fatalf("Failed to create grid output file %s: %v", outputFilename, err)
	}
	defer outFile.Close()

	if err = png.Encode(outFile, gridImage); err != nil {
		log.Fatalf("Failed to encode grid PNG: %v", err)
	}
	fmt.Printf("\nSuccessfully created grid QR code %s.\n", outputFilename)
}

func createAnimatedPNG(images []image.Image, outputFilename string, dataLen int, delayMs int) {
	fmt.Printf("Input is %d bytes, generating an animated PNG with %d frames.\n", dataLen, len(images))
	outFile, err := os.Create(outputFilename)
	if err != nil {
		log.Fatalf("Failed to create output file %s: %v", outputFilename, err)
	}
	defer outFile.Close()

	a := apng.APNG{Frames: []apng.Frame{}}
	for i, img := range images {
		a.Frames = append(a.Frames, apng.Frame{
			Image:            img,
			DelayNumerator:   uint16(delayMs),
			DelayDenominator: 1000, // to convert milliseconds to seconds
		})
		fmt.Printf("Processing frame %d...\n", i+1)
	}

	if err = apng.Encode(outFile, a); err != nil {
		log.Fatalf("Failed to encode animated PNG: %v", err)
	}
	fmt.Printf("\nSuccessfully created animated QR code %s.\n", outputFilename)
}

func encodeMode(format string, chunkSize int, delay int, inputFilename string) {
	// 3. Read the entire input file into memory.
	data, err := os.ReadFile(inputFilename)
	if err != nil {
		log.Fatalf("Failed to read input file %s: %v", inputFilename, err)
	}

	if len(data) == 0 {
		log.Fatalf("Input file %s is empty.", inputFilename)
	}

	// 4. Generate QR codes for each chunk and store them in memory as images.
	var images []image.Image
	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunk := data[i:end]

		// Generate the QR code as a PNG in a byte buffer.
		pngData, err := qrcode.Encode(string(chunk), qrcode.Medium, pngSize)
		if err != nil {
			log.Fatalf("Failed to generate QR code for chunk starting at byte %d: %v", i, err)
		}

		// Decode the in-memory PNG into an image.Image object.
		img, err := png.Decode(bytes.NewReader(pngData))
		if err != nil {
			log.Fatalf("Failed to decode generated QR code PNG for chunk starting at byte %d: %v", i, err)
		}

		images = append(images, img)
	}

	// 5. Determine the single output filename.
	outputFilename := "qr_" + filepath.Base(inputFilename) + ".png"

	// 6. Write the output file based on the number of images and the format flag.
	if len(images) > 1 {
		// Add start (red) and end (blue) marker frames.
		fmt.Println("Adding start and end marker frames.")
		redImg := image.NewRGBA(image.Rect(0, 0, pngSize, pngSize))
		draw.Draw(redImg, redImg.Bounds(), &image.Uniform{C: color.RGBA{R: 0xff, A: 0xff}}, image.Point{}, draw.Src)

		blueImg := image.NewRGBA(image.Rect(0, 0, pngSize, pngSize))
		draw.Draw(blueImg, blueImg.Bounds(), &image.Uniform{C: color.RGBA{B: 0xff, A: 0xff}}, image.Point{}, draw.Src)

		// Prepend red image and append blue image.
		images = append([]image.Image{redImg}, images...)
		images = append(images, blueImg)

		switch format {
		case "apng":
			createAnimatedPNG(images, outputFilename, len(data), delay)
		case "grid":
			createGridPNG(images, outputFilename)
		default:
			log.Fatalf("Invalid format '%s'. Please use 'apng' or 'grid'.", format)
		}
	} else {
		// If there's only one image, write a single static PNG.
		outFile, err := os.Create(outputFilename)
		if err != nil {
			log.Fatalf("Failed to create output file %s: %v", outputFilename, err)
		}
		defer outFile.Close()

		if err = png.Encode(outFile, images[0]); err != nil {
			log.Fatalf("Failed to write single QR code to file: %v", err)
		}
		fmt.Printf("Successfully created QR code file %s from %s.\n", outputFilename, inputFilename)
	}
}

func isSolidColor(img image.Image, c color.RGBA) bool {
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, b2, a := img.At(x, y).RGBA()
			// Convert to 8-bit
			if uint8(r>>8) != c.R || uint8(g>>8) != c.G || uint8(b2>>8) != c.B || uint8(a>>8) != c.A {
				return false
			}
		}
	}
	return true
}

func decodeSingleQR(img image.Image) ([]byte, error) {
	symbols, err := goqr.Recognize(img)
	if err != nil {
		return nil, fmt.Errorf("failed to recognize QR: %w", err)
	}
	if len(symbols) == 0 {
		return nil, errors.New("no QR code found")
	}
	// If more than one, pick the one with the longest payload to reduce chance of picking a small artifact.
	idx := 0
	maxLen := len(symbols[0].Payload)
	for i := 1; i < len(symbols); i++ {
		if l := len(symbols[i].Payload); l > maxLen {
			maxLen = l
			idx = i
		}
	}
	return symbols[idx].Payload, nil
}

func decodeAPNG(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	a, err := apng.DecodeAll(f)
	if err != nil {
		return nil, fmt.Errorf("not an APNG or failed to decode APNG: %w", err)
	}

	var data []byte
	for _, fr := range a.Frames {
		img := fr.Image
		// Skip marker frames (solid red or solid blue with full alpha)
		if isSolidColor(img, color.RGBA{R: 0xff, A: 0xff}) || isSolidColor(img, color.RGBA{B: 0xff, A: 0xff}) {
			continue
		}
		b, err := decodeSingleQR(img)
		if err != nil {
			return nil, fmt.Errorf("failed to decode frame: %w", err)
		}
		data = append(data, b...)
	}
	if len(data) == 0 {
		return nil, errors.New("no data decoded from APNG frames")
	}
	return data, nil
}

func isWhite(pxR, pxG, pxB, pxA uint32) bool {
	return uint8(pxR>>8) == 0xff && uint8(pxG>>8) == 0xff && uint8(pxB>>8) == 0xff && uint8(pxA>>8) == 0xff
}

func decodeGridOrSinglePNG(path string) ([]byte, error) {
	// Open as basic PNG image (single frame).
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("failed to decode PNG: %w", err)
	}

	// Try grid extraction first using known layout (white bg, padding = gridPadding).
	b := img.Bounds()
	bg := img.At(b.Min.X, b.Min.Y)
	r, g, bl, a := bg.RGBA()
	isWhiteBG := isWhite(r, g, bl, a)

	if isWhiteBG {
		// Find the first non-white pixel within a small window after the expected padding.
		startY := b.Min.Y + gridPadding
		startX := b.Min.X + gridPadding
		found := false
		var qrStartX, qrStartY int
		maxProbe := gridPadding + 8 // probe within padding + small tolerance
		for yy := startY; yy < startY+maxProbe && yy < b.Max.Y; yy++ {
			for xx := startX; xx < startX+maxProbe && xx < b.Max.X; xx++ {
				rr, gg, bb, aa := img.At(xx, yy).RGBA()
				if !isWhite(rr, gg, bb, aa) {
					qrStartX = xx
					qrStartY = yy
					found = true
					break
				}
			}
			if found {
				break
			}
		}

		if found {
			// Measure QR width: from qrStartX scan right until we see a run of whites > tolerance.
			whiteRun := 0
			qrW := 0
			tolerance := 3
			yScan := qrStartY
			for x := qrStartX; x < b.Max.X; x++ {
				rr, gg, bb, aa := img.At(x, yScan).RGBA()
				if isWhite(rr, gg, bb, aa) {
					whiteRun++
					if whiteRun > tolerance {
						qrW = x - qrStartX - whiteRun + 1
						break
					}
				} else {
					whiteRun = 0
				}
			}
			if qrW <= 0 {
				// Fallback: approximate width up to next padding
				qrW = pngSize // safe upper bound; will be bounded by image later
			}

			// Measure QR height similarly.
			whiteRun = 0
			qrH := 0
			xScan := qrStartX
			for y := qrStartY; y < b.Max.Y; y++ {
				rr, gg, bb, aa := img.At(xScan, y).RGBA()
				if isWhite(rr, gg, bb, aa) {
					whiteRun++
					if whiteRun > tolerance {
						qrH = y - qrStartY - whiteRun + 1
						break
					}
				} else {
					whiteRun = 0
				}
			}
			if qrH <= 0 {
				qrH = pngSize
			}

			// Clamp qrW/qrH so we don't go out of bounds and also ensure > 0
			if qrW <= 0 || qrH <= 0 {
				// If still invalid, fall back to single QR decode below.
			} else {
				// Iterate tiles row-major using measured qrW/qrH and known padding.
				var payload []byte
				// Use SubImage
				sub, ok := img.(interface {
					SubImage(r image.Rectangle) image.Image
				})
				if !ok {
					return nil, errors.New("image type does not support SubImage")
				}

				// Number of columns/rows estimated from canvas size.
				cols := 0
				for x := b.Min.X + gridPadding; x+qrW <= b.Max.X; x += qrW + gridPadding {
					cols++
				}
				rows := 0
				for y := b.Min.Y + gridPadding; y+qrH <= b.Max.Y; y += qrH + gridPadding {
					rows++
				}

				// Decode each tile; stop a row when decoding fails entirely (assume trailing empties).
				for row := 0; row < rows; row++ {
					yTop := b.Min.Y + gridPadding + row*(qrH+gridPadding)
					rowHasData := false
					for col := 0; col < cols; col++ {
						xLeft := b.Min.X + gridPadding + col*(qrW+gridPadding)
						tileRect := image.Rect(xLeft, yTop, min(xLeft+qrW, b.Max.X), min(yTop+qrH, b.Max.Y))
						qrImg := sub.SubImage(tileRect)

						// Skip solid marker tiles if present
						if isSolidColor(qrImg, color.RGBA{R: 0xff, A: 0xff}) || isSolidColor(qrImg, color.RGBA{B: 0xff, A: 0xff}) {
							continue
						}

						data, err := decodeSingleQR(qrImg)
						if err != nil {
							// If we fail on the first column, consider grid ended after previous rows.
							if col == 0 {
								// If we've already collected any data, end all processing.
								if len(payload) > 0 {
									return payload, nil
								}
								// else continue trying single QR fallback below
								rows = 0
								cols = 0
								break
							}
							// Otherwise stop the current row.
							break
						}
						if len(data) > 0 {
							rowHasData = true
							payload = append(payload, data...)
						}
					}
					if !rowHasData && len(payload) > 0 {
						// Assume we've reached the bottom of the grid.
						return payload, nil
					}
				}
				if len(payload) > 0 {
					return payload, nil
				}
			}
		}
	}

	// Fallback: treat as a single QR
	return decodeSingleQR(img)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func decodeMode(inputPath string, outPath string) {
	// First attempt APNG decode (multi-frame).
	data, err := decodeAPNG(inputPath)
	if err != nil {
		// If APNG failed, attempt single/grid PNG decode.
		data, err = decodeGridOrSinglePNG(inputPath)
		if err != nil {
			log.Fatalf("Decode failed: %v", err)
		}
		// success with grid/single
		writeDecoded(data, outPath)
		return
	}
	// success with APNG
	writeDecoded(data, outPath)
}

func writeDecoded(data []byte, outPath string) {
	var w io.Writer
	if outPath == "" || outPath == "-" {
		w = os.Stdout
	} else {
		f, err := os.Create(outPath)
		if err != nil {
			log.Fatalf("Failed to create output file %s: %v", outPath, err)
		}
		defer f.Close()
		w = f
	}
	if _, err := w.Write(data); err != nil {
		log.Fatalf("Failed to write decoded data: %v", err)
	}
	if outPath != "" && outPath != "-" {
		fmt.Printf("Decoded %d bytes to %s\n", len(data), outPath)
	}
}

func main() {
	// Modes: encode (default) and decode
	mode := flag.String("mode", "encode", "Mode of operation: 'encode' to create QR images, 'decode' to read data from QR images.")
	format := flag.String("format", "apng", "Output format for multiple QR codes: 'apng' for animated PNG or 'grid' for a single grid image. (encode mode only)")
	chunkSize := flag.Int("chunksize", 2048, "The size of each data chunk to be encoded in a single QR code frame. (encode mode only)")
	delay := flag.Int("delay", 1000, "The delay between frames in milliseconds for animated PNGs. (encode mode only)")
	outPath := flag.String("out", "", "Output path for decoded data; use '-' or empty for stdout. (decode mode only)")
	flag.Parse()

	switch *mode {
	case "encode":
		// Validate chunk size.
		if *chunkSize <= 0 {
			log.Fatalf("Error: chunksize must be a positive number.")
		}
		// QR codes can hold up to 2953 bytes with the lowest error correction.
		if *chunkSize > 2953 {
			log.Printf("Warning: chunksize %d is larger than the maximum capacity (2953 bytes) of a QR code. Encoding may fail.", *chunkSize)
		}

		// Validate delay.
		if *delay <= 0 {
			log.Fatalf("Error: delay must be a positive number.")
		}
		if *delay > 65535 {
			log.Fatalf("Error: delay cannot be greater than 65535 milliseconds.")
		}

		// Require exactly one input file.
		if len(flag.Args()) != 1 {
			fmt.Fprintf(os.Stderr, "Usage (encode): %s -mode encode [flags] <input-file>\n", os.Args[0])
			flag.PrintDefaults()
			os.Exit(1)
		}
		inputFilename := flag.Arg(0)
		encodeMode(*format, *chunkSize, *delay, inputFilename)

	case "decode":
		// Require exactly one input file (the PNG/APNG to decode).
		if len(flag.Args()) != 1 {
			fmt.Fprintf(os.Stderr, "Usage (decode): %s -mode decode [flags] <input-image.png>\n", os.Args[0])
			flag.PrintDefaults()
			os.Exit(1)
		}
		inputImage := flag.Arg(0)
		decodeMode(inputImage, *outPath)

	default:
		log.Fatalf("Invalid mode '%s'. Please use 'encode' or 'decode'.", *mode)
	}
}
