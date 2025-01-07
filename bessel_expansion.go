package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"sync"

	"github.com/tdanieljr/gomat"
)

func expansion_eval(rs, zs, betas []float64, coeffs []complex128, wavenumber float64) gomat.Matrix[complex128] {
	integral := gomat.Init[complex128](len(rs), len(zs))
	dbeta := complex(betas[1]-betas[0], 0)

	type result struct {
		i, j  int
		value complex128
	}

	numWorkers := 6
	var wg sync.WaitGroup

	count := len(betas) - 1
	// Split work into chunks
	chunkSize := count / numWorkers
	if count%numWorkers != 0 {
		chunkSize++ // If count isn't perfectly divisible by numWorkers, handle the remainder
	}
	res := make(chan result, len(rs)*len(zs)*numWorkers)

	for worker := range numWorkers {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * chunkSize
			end := start + chunkSize
			if end > count {
				end = count
			}

			// Partial sums of Trapezoidal integration of Bessel expansion
			for i, r := range rs {
				for j, z := range zs {
					t := complex(0, 0)
					for k := start; k < end; k++ {
						t1 := coeffs[k] * complex(math.J0(wavenumber*math.Sin(betas[k])*r), 0) * cmplx.Exp(complex(0, 1)*complex(wavenumber*math.Cos(betas[k])*z, 0))
						t2 := coeffs[k+1] * complex(math.J0(wavenumber*math.Sin(betas[k+1])*r), 0) * cmplx.Exp(complex(0, 1)*complex(wavenumber*math.Cos(betas[k+1])*z, 0))
						t += 0.5 * dbeta * (t1 + t2)
					}
					res <- result{i: i, j: j, value: t}
				}
			}
		}(worker)
	}

	wg.Wait()
	close(res)
	for r := range res {
		integral.Set(r.i, r.j, integral.Get(r.i, r.j)+r.value)
	}
	return integral
}

func coefficients(rs, betas []float64, profile []complex128, wavenumber float64) []complex128 {
	integrand := gomat.Init[complex128](len(betas), len(rs))
	for i, r := range rs {
		for j, b := range betas {
			integrand.Set(j, i, profile[i]*complex(math.J0(wavenumber*math.Sin(b)*r)*r, 0))
		}
	}

	coeffs := integrand.IntegrateRows(gomat.RealToComplex(rs))
	for i := range coeffs {
		coeffs[i] *= complex(wavenumber*math.Cos(betas[i])*wavenumber*math.Sin(betas[i]), 0)
	}
	return coeffs
}

func focused_profile(thetas []float64, focal_length, alpha, wavenumber float64) []complex128 {
	extra_phase := make([]float64, len(thetas))
	for i, val := range thetas {
		extra_phase[i] = focal_length * (1 - math.Cos(alpha)/math.Cos(val))
	}
	out := make([]complex128, len(thetas))
	for i, ep := range extra_phase {
		out[i] = cmplx.Exp(complex(0, 1)*complex(wavenumber, 0)*complex(ep, 0)) * complex(math.Cos(thetas[i])/math.Cos(alpha), 0)
	}
	return out
}

func beta_sampling(max_z float64, wavenumber, radius float64) float64 {
	if math.Pi/(2*wavenumber*max_z) < math.Pi/(2*wavenumber*radius) {
		return math.Pi / (2 * wavenumber * max_z)
	}
	return math.Pi / (2 * wavenumber * radius)
}

func main() {
	// h104,
	radius := 32.0 / 1000.0
	focal_length := 63.2 / 1000.0
	wavenumber := 2 * math.Pi * 1.5e6 / 1485
	alpha := math.Asin(radius / focal_length)    // angle subtended by the transducer
	hmax := focal_length * (1 - math.Cos(alpha)) // depth of the transducer
	wavelength := (2 * math.Pi) / wavenumber

	rs := gomat.Linspace(0, radius, wavelength/8)
	thetas := make([]float64, len(rs))
	for i, r := range rs {
		thetas[i] = math.Atan(r / (focal_length - hmax))
	}

	zs := gomat.Linspace(0, focal_length*2, wavelength/8)
	dbeta := beta_sampling(zs[len(zs)-1], wavenumber, radius)
	betas := gomat.Linspace(0, math.Pi/2, dbeta)
	profile := focused_profile(thetas, focal_length, alpha, wavenumber)
	coeffs := coefficients(rs, betas, profile, wavenumber)
	expansion := expansion_eval(rs, zs, betas, coeffs, wavenumber)
	fmt.Println(expansion.Get(0, 0))
	// fmt.Println(len(expansion))
}
