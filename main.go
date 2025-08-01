package main

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/kettek/apng"
	"github.com/skip2/go-qrcode"
)

// The maximum number of bytes to encode in a single QR code.
// QR codes can hold up to 2953 bytes with the lowest error correction.
// We use a smaller chunk size to be safe and allow for higher error correction levels.
const chunkSize = 2048

// The size of the generated PNG image in pixels.
const pngSize = 512

func main() {
	// 1. Check for a single command-line argument for the input file.
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <input-file>\n", os.Args[0])
		os.Exit(1)
	}
	inputFilename := os.Args[1]

	// 2. Read the entire input file into memory.
	data, err := os.ReadFile(inputFilename)
	if err != nil {
		log.Fatalf("Failed to read input file %s: %v", inputFilename, err)
	}

	if len(data) == 0 {
		log.Fatalf("Input file %s is empty.", inputFilename)
	}

	// 3. Generate QR codes for each chunk and store them in memory as images.
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

	// 4. Determine the single output filename.
	outputFilename := "qr_" + filepath.Base(inputFilename) + ".png"

	// 5. Write the output file.
	if len(images) > 1 {
		// If there are multiple images, create an animated PNG.
		fmt.Printf("Input is %d bytes, generating an animated PNG with %d frames.\n", len(data), len(images))
		outFile, err := os.Create(outputFilename)
		if err != nil {
			log.Fatalf("Failed to create output file %s: %v", outputFilename, err)
		}
		defer outFile.Close()

		a := apng.APNG{Frames: []apng.Frame{}}
		for i, img := range images {
			a.Frames = append(a.Frames, apng.Frame{
				Image:            img,
				DelayNumerator:   1, // 1 second delay per frame
				DelayDenominator: 1,
			})
			fmt.Printf("Processing frame %d...\n", i+1)
		}

		if err = apng.Encode(outFile, a); err != nil {
			log.Fatalf("Failed to encode animated PNG: %v", err)
		}
		fmt.Printf("\nSuccessfully created animated QR code %s.\n", outputFilename)

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
