package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/kettek/apng"
	"github.com/liyue201/goqr"
	"github.com/skip2/go-qrcode"
)

// The size of the generated PNG image in pixels.
const pngSize = 512

// Padding used in grid generation between QR codes and around the border.
const gridPadding = 20

// Global flag to control debug output
var debugMode bool

// decodeModeFlag controls forced decode mode: "auto" (default), "apng", "grid", "photo", "single".
var decodeModeFlag string

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

// Strict solid color check (kept for exact programmatic grids/APNGs).
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

// nearColor returns true if the pixel is close to the target color within tolerance.
func nearColor(px color.Color, target color.RGBA, tol uint8) bool {
	r, g, b, a := px.RGBA()
	r8 := uint8(r >> 8)
	g8 := uint8(g >> 8)
	b8 := uint8(b >> 8)
	a8 := uint8(a >> 8)
	// require reasonable opacity to avoid backgrounds
	if a8 < 200 {
		return false
	}
	absDiff := func(a, b uint8) uint8 {
		if a > b {
			return a - b
		}
		return b - a
	}
	return absDiff(r8, target.R) <= tol &&
		absDiff(g8, target.G) <= tol &&
		absDiff(b8, target.B) <= tol
}

// isMostlyColor samples the image on a coarse grid and returns true if the ratio of pixels
// near the target color is >= minRatio.
func isMostlyColor(img image.Image, target color.RGBA, tol uint8, minRatio float64) bool {
	b := img.Bounds()
	width := b.Dx()
	height := b.Dy()
	if width == 0 || height == 0 {
		return false
	}

	// sample every N pixels, scaling with size
	stepX := max(4, width/64)
	stepY := max(4, height/64)
	var total, hits int
	for y := b.Min.Y; y < b.Max.Y; y += stepY {
		for x := b.Min.X; x < b.Max.X; x += stepX {
			total++
			if nearColor(img.At(x, y), target, tol) {
				hits++
			}
		}
	}
	if total == 0 {
		return false
	}
	return float64(hits)/float64(total) >= minRatio
}

// decodeSingleQR returns the payload from the best QR found in the image.
func decodeSingleQR(img image.Image) ([]byte, error) {
	symbols, err := goqr.Recognize(img)
	if err != nil {
		return nil, fmt.Errorf("failed to recognize QR: %w", err)
	}
	if len(symbols) == 0 {
		return nil, errors.New("no QR code found")
	}
	// Pick the one with the longest payload.
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

// decodeAPNG decodes APNG and returns data, number of QR frames decoded, and number of non-marker frames.
func decodeAPNG(path string) ([]byte, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	a, err := apng.DecodeAll(f)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("not an APNG or failed to decode APNG: %w", err)
	}

	var data []byte
	qrCount := 0
	nonMarkerFrames := 0
	for _, fr := range a.Frames {
		img := fr.Image
		// Robustly skip marker frames (mostly red or mostly blue)
		if isMostlyColor(img, color.RGBA{R: 0xff, G: 0x00, B: 0x00, A: 0xff}, 32, 0.85) ||
			isMostlyColor(img, color.RGBA{R: 0x00, G: 0x00, B: 0xff, A: 0xff}, 32, 0.85) ||
			isSolidColor(img, color.RGBA{R: 0xff, A: 0xff}) || // keep strict as fast-path
			isSolidColor(img, color.RGBA{B: 0xff, A: 0xff}) {
			continue
		}
		nonMarkerFrames++
		b, err := decodeSingleQR(img)
		if err != nil {
			// If a non-marker frame doesn't decode, keep scanning, but don't count it.
			continue
		}
		qrCount++
		data = append(data, b...)
	}
	if qrCount == 0 {
		return nil, 0, nonMarkerFrames, errors.New("no data decoded from APNG frames")
	}
	return data, qrCount, nonMarkerFrames, nil
}

func isWhite(pxR, pxG, pxB, pxA uint32) bool {
	return uint8(pxR>>8) == 0xff && uint8(pxG>>8) == 0xff && uint8(pxB>>8) == 0xff && uint8(pxA>>8) == 0xff
}

// rotate90 rotates an image 90 degrees clockwise.
func rotate90(src image.Image) image.Image {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dy(), b.Dx()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(b.Max.Y-1-y, x-b.Min.X, src.At(x, y))
		}
	}
	return dst
}

// rotate180 rotates an image 180 degrees.
func rotate180(src image.Image) image.Image {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(b.Max.X-1-x, b.Max.Y-1-y, src.At(x, y))
		}
	}
	return dst
}

// rotate270 rotates an image 270 degrees clockwise (90 ccw).
func rotate270(src image.Image) image.Image {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dy(), b.Dx()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(y-b.Min.Y, b.Max.X-1-x, src.At(x, y))
		}
	}
	return dst
}

// boostContrast applies a simple linear contrast stretch around 0.5 to increase QR contrast.
func boostContrast(src image.Image) image.Image {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, a := src.At(x, y).RGBA()
			// normalize to [0..255]
			r8 := uint8(r >> 8)
			g8 := uint8(g >> 8)
			b8 := uint8(bl >> 8)
			// simple contrast around 128 with factor 1.5
			apply := func(v uint8) uint8 {
				f := float64(v)
				f = (f-128)*1.5 + 128
				if f < 0 {
					f = 0
				}
				if f > 255 {
					f = 255
				}
				return uint8(f)
			}
			dst.SetRGBA(x, y, color.RGBA{
				R: apply(r8),
				G: apply(g8),
				B: apply(b8),
				A: uint8(a >> 8),
			})
		}
	}
	return dst
}

// grayscale converts the image to grayscale (luminance).
func grayscale(src image.Image) image.Image {
	b := src.Bounds()
	dst := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
	return dst
}

// cropMargins tries to detect and crop uniform background margins to tighten the content area.
// It assumes background is close to white and content contains darker pixels.
func cropMargins(src image.Image) image.Image {
	b := src.Bounds()
	minX, minY := b.Max.X, b.Max.Y
	maxX, maxY := b.Min.X, b.Min.Y

	isContent := func(c color.Color) bool {
		r, g, bl, _ := c.RGBA()
		// consider not white if any channel < 240
		return (uint8(r>>8) < 240) || (uint8(g>>8) < 240) || (uint8(bl>>8) < 240)
	}

	stepX := max(1, b.Dx()/400)
	stepY := max(1, b.Dy()/400)

	for y := b.Min.Y; y < b.Max.Y; y += stepY {
		for x := b.Min.X; x < b.Max.X; x += stepX {
			if isContent(src.At(x, y)) {
				if x < minX {
					minX = x
				}
				if y < minY {
					minY = y
				}
				if x > maxX {
					maxX = x
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	// If we failed to find content, return original
	if minX >= maxX || minY >= maxY {
		return src
	}
	// Expand slightly
	padX := max(4, b.Dx()/100)
	padY := max(4, b.Dy()/100)
	minX = max(b.Min.X, minX-padX)
	minY = max(b.Min.Y, minY-padY)
	maxX = min(b.Max.X, maxX+padX)
	maxY = min(b.Max.Y, maxY+padY)

	if s, ok := src.(interface{ SubImage(r image.Rectangle) image.Image }); ok {
		return s.SubImage(image.Rect(minX, minY, maxX, maxY))
	}
	// Fallback: copy region
	dst := image.NewRGBA(image.Rect(0, 0, maxX-minX, maxY-minY))
	draw.Draw(dst, dst.Bounds(), src, image.Point{X: minX, Y: minY}, draw.Src)
	return dst
}

// detectAllQRCodes tries to detect QR codes in the given image with multiple attempts.
// For photo inputs (grid photos), we aggressively preprocess and attempt multiple rotations.
func detectAllQRCodes(img image.Image) ([]*goqr.QRData, error) {
	var variants []image.Image

	// Base variants
	variants = append(variants, img)
	variants = append(variants, cropMargins(img))
	variants = append(variants, grayscale(img))
	variants = append(variants, boostContrast(img))

	// Rotations for robustness
	makeRotations := func(im image.Image) {
		r90 := rotate90(im)
		r180 := rotate180(im)
		r270 := rotate270(im)
		variants = append(variants, r90, r180, r270)
	}
	// Create rotations for key variants
	makeRotations(img)
	makeRotations(cropMargins(img))
	makeRotations(boostContrast(img))

	// Try each variant until one yields any symbols
	found := []*goqr.QRData(nil)
	for _, im := range variants {
		syms, err := goqr.Recognize(im)
		if err != nil {
			continue
		}
		if len(syms) > 0 {
			found = syms
			break
		}
	}

	if len(found) == 0 {
		return nil, errors.New("no QR codes detected")
	}
	return found, nil
}

// orderSymbolsRowMajor orders multiple QR detections by their centroid:
// top-to-bottom, left-to-right with a vertical tolerance to form rows.
// Our goqr version lacks geometry; as a fallback we keep the order as returned.
func orderSymbolsRowMajor(symbols []*goqr.QRData) []*goqr.QRData {
	// Prefer longer payload first to reduce noise if multiple are detected.
	sort.SliceStable(symbols, func(i, j int) bool {
		return len(symbols[i].Payload) > len(symbols[j].Payload)
	})
	return symbols
}

func decodePhotoLike(img image.Image) ([]byte, int, error) {
	symbols, err := detectAllQRCodes(img)
	if err != nil {
		return nil, 0, err
	}
	if len(symbols) == 1 {
		return symbols[0].Payload, 1, nil
	}
	ordered := orderSymbolsRowMajor(symbols)
	var out []byte
	count := 0
	for _, s := range ordered {
		if len(s.Payload) == 0 {
			continue
		}
		out = append(out, s.Payload...)
		count++
	}
	if count == 0 {
		return nil, 0, errors.New("no payloads found after ordering")
	}
	return out, count, nil
}

// drawSymbolsDebug outlines each detected symbol with a green rectangle if geometry is available.
// Our library version lacks per-symbol geometry, so we draw a coarse grid overlay as a proxy:
// we try to infer a regular grid layout and draw rectangles at the estimated tile positions.
func drawSymbolsDebug(inputPath string, src image.Image, symbols []*goqr.QRData) error {
	// Attempt to infer a grid overlay similar to decodeGridHeuristic
	b := src.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, src, b.Min, draw.Src)
	green := color.RGBA{R: 0x00, G: 0xff, B: 0x00, A: 0xff}

	// Try detect white background to infer grid padding/tiles; otherwise draw whole image box.
	bg := src.At(b.Min.X, b.Min.Y)
	r, g, bl, a := bg.RGBA()
	if !(uint8(r>>8) == 0xff && uint8(g>>8) == 0xff && uint8(bl>>8) == 0xff && uint8(a>>8) == 0xff) {
		drawRect(rgba, b, green, max(2, b.Dx()/200))
		out := debugOutputPath(inputPath)
		f, err := os.Create(out)
		if err != nil {
			return err
		}
		defer f.Close()
		return png.Encode(f, rgba)
	}

	// Reuse the probing logic from grid heuristic to estimate first tile size.
	startY := b.Min.Y + gridPadding
	startX := b.Min.X + gridPadding
	found := false
	var qrStartX, qrStartY int
	maxProbe := gridPadding + 16
	for yy := startY; yy < startY+maxProbe && yy < b.Max.Y; yy++ {
		for xx := startX; xx < startX+maxProbe && xx < b.Max.X; xx++ {
			rr, gg, bb, aa := src.At(xx, yy).RGBA()
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

	if !found {
		drawRect(rgba, b, green, max(2, b.Dx()/200))
		out := debugOutputPath(inputPath)
		f, err := os.Create(out)
		if err != nil {
			return err
		}
		defer f.Close()
		return png.Encode(f, rgba)
	}

	whiteRun := 0
	qrW := 0
	tolerance := 5
	yScan := qrStartY
	for x := qrStartX; x < b.Max.X; x++ {
		rr, gg, bb, aa := src.At(x, yScan).RGBA()
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
		qrW = max(32, b.Dx()/8)
	}
	whiteRun = 0
	qrH := 0
	xScan := qrStartX
	for y := qrStartY; y < b.Max.Y; y++ {
		rr, gg, bb, aa := src.At(xScan, y).RGBA()
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
		qrH = qrW
	}

	// Draw an overlay grid based on estimated tile size and known padding
	cols := 0
	for x := b.Min.X + gridPadding; x+qrW <= b.Max.X; x += qrW + gridPadding {
		cols++
	}
	rows := 0
	for y := b.Min.Y + gridPadding; y+qrH <= b.Max.Y; y += qrH + gridPadding {
		rows++
	}

	thickness := max(2, min(qrW, qrH)/40)
	for row := 0; row < rows; row++ {
		yTop := b.Min.Y + gridPadding + row*(qrH+gridPadding)
		for col := 0; col < cols; col++ {
			xLeft := b.Min.X + gridPadding + col*(qrW+gridPadding)
			tileRect := image.Rect(xLeft, yTop, min(xLeft+qrW, b.Max.X), min(yTop+qrH, b.Max.Y))
			drawRect(rgba, tileRect, green, thickness)
		}
	}

	out := debugOutputPath(inputPath)
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, rgba)
}

func decodeStaticImageFile(path string) ([]byte, string, int, error) {
	ext := strings.ToLower(filepath.Ext(path))
	// Open file
	f, err := os.Open(path)
	if err != nil {
		return nil, "", 0, err
	}
	defer f.Close()

	var img image.Image
	switch ext {
	case ".png":
		// Try decode as PNG first (may be static PNG). APNG is handled elsewhere.
		img, err = png.Decode(f)
	case ".jpg", ".jpeg":
		img, err = jpeg.Decode(f)
	default:
		// Try PNG, then JPEG as a fallback based on sniffer
		if img, err = png.Decode(f); err != nil {
			// Re-open to reset reader
			_ = f.Close()
			f2, err2 := os.Open(path)
			if err2 != nil {
				return nil, "", 0, err2
			}
			defer f2.Close()
			img, err = jpeg.Decode(f2)
		}
	}
	if err != nil {
		return nil, "", 0, fmt.Errorf("failed to decode image %s: %w", path, err)
	}

	// Photo-like robust multi-QR decode (assumes a photographed grid)
	if symbols, err := detectAllQRCodes(img); err == nil && len(symbols) > 0 {
		ordered := orderSymbolsRowMajor(symbols)
		var out []byte
		valid := 0
		for _, s := range ordered {
			if len(s.Payload) == 0 {
				continue
			}
			out = append(out, s.Payload...)
			valid++
		}
		if valid > 0 {
			if debugMode {
				if err := drawSymbolsDebug(path, img, ordered); err != nil {
					log.Printf("debug: failed to save photo debug image: %v", err)
				}
			}
			return out, "photo", valid, nil
		}
	}

	// If PNG, try exact grid heuristic as last resort (for generated grids).
	if ext == ".png" {
		// Re-open to decode again for grid heuristic path which expects png.Image for SubImage etc.
		ff, err := os.Open(path)
		if err != nil {
			return nil, "", 0, err
		}
		defer ff.Close()
		pimg, err := png.Decode(ff)
		if err != nil {
			return nil, "", 0, fmt.Errorf("failed to decode PNG for grid decode: %w", err)
		}
		data, count, err := decodeGridHeuristic(pimg)
		if err == nil {
			// For grid heuristic, draw rectangles where we decoded tiles.
			if debugMode {
				if err := saveGridDebugImage(path, pimg, count); err != nil {
					log.Printf("debug: failed to save grid debug image: %v", err)
				}
			}
			return data, "grid", count, nil
		}
	}

	// As a final fallback, try single QR on the whole image.
	if payload, err := decodeSingleQR(img); err == nil {
		// Single QR: draw a border around the whole image as a coarse indication.
		if debugMode {
			if err := saveWholeImageBox(path, img); err != nil {
				log.Printf("debug: failed to save single debug image: %v", err)
			}
		}
		return payload, "single", 1, nil
	}

	// For JPEG, no grid heuristic; return the original error.
	return nil, "", 0, errors.New("failed to decode QR(s) from image")
}

// The previous grid heuristic, refactored to accept an already-decoded image.
// Returns data and number of tiles decoded.
func decodeGridHeuristic(img image.Image) ([]byte, int, error) {
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

			// Iterate tiles row-major using measured qrW/qrH and known padding.
			var payload []byte
			qrCount := 0
			// Use SubImage
			sub, ok := img.(interface {
				SubImage(r image.Rectangle) image.Image
			})
			if !ok {
				return nil, 0, errors.New("image type does not support SubImage")
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

					// Robustly skip marker tiles (mostly red or mostly blue)
					if isMostlyColor(qrImg, color.RGBA{R: 0xff, G: 0x00, B: 0x00, A: 0xff}, 32, 0.85) ||
						isMostlyColor(qrImg, color.RGBA{R: 0x00, G: 0x00, B: 0xff, A: 0xff}, 32, 0.85) ||
						isSolidColor(qrImg, color.RGBA{R: 0xff, A: 0xff}) ||
						isSolidColor(qrImg, color.RGBA{B: 0xff, A: 0xff}) {
						continue
					}

					data, err := decodeSingleQR(qrImg)
					if err != nil {
						// If we fail on the first column, consider grid ended after previous rows.
						if col == 0 {
							// If we've already collected any data, end all processing.
							if len(payload) > 0 {
								return payload, qrCount, nil
							}
							// else give up on grid heuristic
							rows = 0
							cols = 0
							break
						}
						// Otherwise stop the current row.
						break
					}
					if len(data) > 0 {
						rowHasData = true
						qrCount++
						payload = append(payload, data...)
					}
				}
				if !rowHasData && len(payload) > 0 {
					// Assume we've reached the bottom of the grid.
					return payload, qrCount, nil
				}
			}
			if len(payload) > 0 {
				return payload, qrCount, nil
			}
		}
	}

	// Fallback: treat as a single QR
	payload, err := decodeSingleQR(img)
	if err != nil {
		return nil, 0, err
	}
	return payload, 1, nil
}

func drawRect(img draw.Image, r image.Rectangle, c color.Color, thickness int) {
	if thickness < 1 {
		thickness = 1
	}
	// top
	for y := r.Min.Y; y < r.Min.Y+thickness; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.Set(x, y, c)
		}
	}
	// bottom
	for y := r.Max.Y - thickness; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.Set(x, y, c)
		}
	}
	// left
	for x := r.Min.X; x < r.Min.X+thickness; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			img.Set(x, y, c)
		}
	}
	// right
	for x := r.Max.X - thickness; x < r.Max.X; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			img.Set(x, y, c)
		}
	}
}

// saveWholeImageBox saves an image with a green border around the whole bounds.
func saveWholeImageBox(inputPath string, src image.Image) error {
	b := src.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, src, b.Min, draw.Src)
	green := color.RGBA{R: 0x00, G: 0xff, B: 0x00, A: 0xff}
	drawRect(rgba, b, green, max(2, b.Dx()/200))
	out := debugOutputPath(inputPath)
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, rgba)
}

// saveGridDebugImage draws green rectangles on grid tiles that decoded successfully.
// Since we do not store tile rectangles during decode, we re-run the grid tiling to draw all candidate tiles,
// marking all tile positions; this provides a visual grid overlay matching the decode heuristic.
func saveGridDebugImage(inputPath string, src image.Image, tileCount int) error {
	b := src.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, src, b.Min, draw.Src)
	green := color.RGBA{R: 0x00, G: 0xff, B: 0x00, A: 0xff}

	// Recreate grid layout similarly to decodeGridHeuristic to determine tile rects.
	bg := src.At(b.Min.X, b.Min.Y)
	r, g, bl, a := bg.RGBA()
	if !isWhite(r, g, bl, a) {
		// If not white background, just draw a whole-image box.
		drawRect(rgba, b, green, max(2, b.Dx()/200))
	} else {
		startY := b.Min.Y + gridPadding
		startX := b.Min.X + gridPadding
		found := false
		var qrStartX, qrStartY int
		maxProbe := gridPadding + 8
		for yy := startY; yy < startY+maxProbe && yy < b.Max.Y; yy++ {
			for xx := startX; xx < startX+maxProbe && xx < b.Max.X; xx++ {
				rr, gg, bb, aa := src.At(xx, yy).RGBA()
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
		if !found {
			drawRect(rgba, b, green, max(2, b.Dx()/200))
		} else {
			// measure widths/heights same as decodeGridHeuristic
			whiteRun := 0
			qrW := 0
			tolerance := 3
			yScan := qrStartY
			for x := qrStartX; x < b.Max.X; x++ {
				rr, gg, bb, aa := src.At(x, yScan).RGBA()
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
				qrW = pngSize
			}
			whiteRun = 0
			qrH := 0
			xScan := qrStartX
			for y := qrStartY; y < b.Max.Y; y++ {
				rr, gg, bb, aa := src.At(xScan, y).RGBA()
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

			cols := 0
			for x := b.Min.X + gridPadding; x+qrW <= b.Max.X; x += qrW + gridPadding {
				cols++
			}
			rows := 0
			for y := b.Min.Y + gridPadding; y+qrH <= b.Max.Y; y += qrH + gridPadding {
				rows++
			}
			thickness := max(2, min(qrW, qrH)/40)
			count := 0
			for row := 0; row < rows; row++ {
				yTop := b.Min.Y + gridPadding + row*(qrH+gridPadding)
				for col := 0; col < cols; col++ {
					xLeft := b.Min.X + gridPadding + col*(qrW+gridPadding)
					tileRect := image.Rect(xLeft, yTop, min(xLeft+qrW, b.Max.X), min(yTop+qrH, b.Max.Y))
					drawRect(rgba, tileRect, green, thickness)
					count++
					if tileCount > 0 && count >= tileCount {
						// Optional: stop after drawing number of decoded tiles.
						// But generally drawing all tiles is more informative.
					}
				}
			}
		}
	}

	out := debugOutputPath(inputPath)
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, rgba)
}

func debugOutputPath(inputPath string) string {
	dir := filepath.Dir(inputPath)
	base := filepath.Base(inputPath)
	ext := filepath.Ext(base)
	name := strings.TrimSuffix(base, ext)
	return filepath.Join(dir, fmt.Sprintf("%s_debug.png", name))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func decodeMode(inputPath string, outPath string) {
	ext := strings.ToLower(filepath.Ext(inputPath))

	// Forced mode handling
	switch decodeModeFlag {
	case "apng":
		if ext != ".png" {
			log.Fatalf("Forced mode=apng requires a .png file")
		}
		if data, qrCount, nonMarker, err := decodeAPNG(inputPath); err == nil && nonMarker >= 2 && qrCount >= 2 {
			fmt.Printf("mode=decode format=apng qrs=%d\n", qrCount)
			writeDecoded(data, outPath)
			return
		} else if err != nil {
			log.Fatalf("APNG decode failed: %v", err)
		} else {
			log.Fatalf("APNG decode did not yield enough frames (nonMarker=%d qrs=%d)", nonMarker, qrCount)
		}
	case "grid":
		// Open PNG and run grid heuristic
		if ext != ".png" {
			log.Fatalf("Forced mode=grid requires a .png file")
		}
		f, err := os.Open(inputPath)
		if err != nil {
			log.Fatalf("Failed to open image: %v", err)
		}
		defer f.Close()
		img, err := png.Decode(f)
		if err != nil {
			log.Fatalf("Failed to decode PNG: %v", err)
		}
		data, count, err := decodeGridHeuristic(img)
		if err != nil {
			log.Fatalf("Grid decode failed: %v", err)
		}
		if debugMode {
			if err := saveGridDebugImage(inputPath, img, count); err != nil {
				log.Printf("debug: failed to save grid debug image: %v", err)
			}
		}
		fmt.Printf("mode=decode format=grid qrs=%d\n", count)
		writeDecoded(data, outPath)
		return
	case "photo":
		// Decode as static image using photo path first
		data, _, count, err := decodeStaticImageFile(inputPath)
		if err != nil {
			log.Fatalf("Photo decode failed: %v", err)
		}
		// Ensure we report as photo (even if fallback picked single); keep logging as photo
		fmt.Printf("mode=decode format=photo qrs=%d\n", count)
		writeDecoded(data, outPath)
		return
	case "single":
		// Force single QR decode for any image type
		// Open via PNG or JPEG
		f, err := os.Open(inputPath)
		if err != nil {
			log.Fatalf("Failed to open image: %v", err)
		}
		defer f.Close()
		ext2 := strings.ToLower(filepath.Ext(inputPath))
		var img image.Image
		if ext2 == ".png" {
			img, err = png.Decode(f)
		} else {
			img, err = jpeg.Decode(f)
		}
		if err != nil {
			log.Fatalf("Failed to decode image: %v", err)
		}
		payload, err := decodeSingleQR(img)
		if err != nil {
			log.Fatalf("Single decode failed: %v", err)
		}
		if debugMode {
			if err := saveWholeImageBox(inputPath, img); err != nil {
				log.Printf("debug: failed to save single debug image: %v", err)
			}
		}
		fmt.Printf("mode=decode format=single qrs=%d\n", 1)
		writeDecoded(payload, outPath)
		return
	case "auto", "":
		// proceed to auto flow below
	default:
		log.Fatalf("Invalid -decode-mode value: %s (valid: auto, apng, grid, photo, single)", decodeModeFlag)
	}

	// Auto-detection flow (existing behavior)
	// For PNGs, try APNG first as before.
	if ext == ".png" {
		if data, qrCount, nonMarker, err := decodeAPNG(inputPath); err == nil && nonMarker >= 2 && qrCount >= 2 {
			// Accept APNG only if there are at least 2 non-marker frames and >=2 decodable QR frames.
			fmt.Printf("mode=decode format=apng qrs=%d\n", qrCount)
			writeDecoded(data, outPath)
			return
		}
		// Otherwise fall through to static image path.
	}

	// Static image decode path supports PNG and JPEG.
	data, fmtDetected, count, err := decodeStaticImageFile(inputPath)
	if err != nil {
		log.Fatalf("Decode failed: %v", err)
	}
	if fmtDetected == "" {
		fmtDetected = "unknown"
	}
	fmt.Printf("mode=decode format=%s qrs=%d\n", fmtDetected, count)
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
	flag.BoolVar(&debugMode, "debug", false, "Enable saving a debug image with green boxes around detected QRs during decode.")
	flag.StringVar(&decodeModeFlag, "decode-mode", "auto", "Force decode mode: 'auto' (default), 'apng', 'grid', 'photo', or 'single'.")
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
		// Require exactly one input file (the PNG/JPG/APNG to decode).
		if len(flag.Args()) != 1 {
			fmt.Fprintf(os.Stderr, "Usage (decode): %s -mode decode [flags] <input-image.(png|jpg|jpeg)>\n", os.Args[0])
			flag.PrintDefaults()
			os.Exit(1)
		}
		inputImage := flag.Arg(0)
		decodeMode(inputImage, *outPath)

	default:
		log.Fatalf("Invalid mode '%s'. Please use 'encode' or 'decode'.", *mode)
	}
}
