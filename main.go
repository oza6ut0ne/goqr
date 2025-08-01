package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.comcom/skip2/go-qrcode"
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

	// 3. Use the input filename as a base for the output QR code files.
	baseOutputFilename := "qr_" + filepath.Base(inputFilename)

	// 4. Split the data into chunks and generate a QR code for each.
	chunkCount := 0
	for i := 0; i < len(data); i += chunkSize {
		chunkCount++
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunk := data[i:end]

		// Suffix the output filename with the chunk number.
		outputFilename := fmt.Sprintf("%s_%d.png", baseOutputFilename, chunkCount)

		// Generate and write the QR code to a PNG file.
		// We use qrcode.Medium for a good balance of data density and error correction.
		// The content is cast to a string, which is safe for binary data in Go.
		err := qrcode.WriteFile(string(chunk), qrcode.Medium, pngSize, outputFilename)
		if err != nil {
			log.Fatalf("Failed to generate QR code for chunk %d: %v", chunkCount, err)
		}

		fmt.Printf("Generated %s for chunk %d (%d bytes)\n", outputFilename, chunkCount, len(chunk))
	}

	fmt.Printf("\nSuccessfully created %d QR code file(s) from %s.\n", chunkCount, inputFilename)
}
