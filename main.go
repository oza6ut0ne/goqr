package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/kettek/apng"
	"github.com/skip2/go-qrcode"
)

// The size of the generated PNG image in pixels.
const pngSize = 512

func createGridPNG(images []image.Image, outputFilename string) {
	if len(images) == 0 {
		log.Println("No images to create a grid from.")
		return
	}

	const padding = 20 // Padding between QR codes and around the border.

	// All QR codes are the same size.
	qrWidth := images[0].Bounds().Dx()
	qrHeight := images[0].Bounds().Dy()

	// Calculate grid dimensions to be as square as possible.
	numImages := len(images)
	cols := int(math.Ceil(math.Sqrt(float64(numImages))))
	rows := int(math.Ceil(float64(numImages) / float64(cols)))

	// Calculate final image dimensions with padding.
	gridWidth := cols*qrWidth + (cols+1)*padding
	gridHeight := rows*qrHeight + (rows+1)*padding

	// Create a new RGBA image to act as the canvas.
	gridImage := image.NewRGBA(image.Rect(0, 0, gridWidth, gridHeight))
	// Fill canvas with a white background.
	draw.Draw(gridImage, gridImage.Bounds(), image.White, image.Point{}, draw.Src)

	// Draw each QR code onto the grid.
	for i, img := range images {
		row := i / cols
		col := i % cols
		// Calculate the top-left corner for this QR code.
		x := padding + col*(qrWidth+padding)
		y := padding + row*(qrHeight+padding)
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

func main() {
	// 1. Define and parse command-line flags.
	format := flag.String("format", "apng", "Output format for multiple QR codes: 'apng' for animated PNG or 'grid' for a single grid image.")
	chunkSize := flag.Int("chunksize", 2048, "The size of each data chunk to be encoded in a single QR code frame.")
	delay := flag.Int("delay", 1000, "The delay between frames in milliseconds for animated PNGs.")
	flag.Parse()

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

	// 2. Check for a single command-line argument for the input file.
	if len(flag.Args()) != 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <input-file>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}
	inputFilename := flag.Arg(0)

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
	for i := 0; i < len(data); i += *chunkSize {
		end := i + *chunkSize
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

		switch *format {
		case "apng":
			createAnimatedPNG(images, outputFilename, len(data), *delay)
		case "grid":
			createGridPNG(images, outputFilename)
		default:
			log.Fatalf("Invalid format '%s'. Please use 'apng' or 'grid'.", *format)
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
