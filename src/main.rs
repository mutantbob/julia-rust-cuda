use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::{cuda_launch, load_kernel_module};
use num_complex::{Complex, Complex32};
use std::fs::File;
use std::io::Write;

pub struct JuliaGrid {
    pub ncols: usize,
    pub nrows: usize,
    pub dx: f32,
    pub dy: f32,
    pub x0: f32,
    pub y0: f32,
}

impl JuliaGrid {
    fn cell_count(&self) -> usize {
        self.nrows * self.ncols
    }

    pub fn xy_for(&self, col: usize, row: usize) -> (f32, f32) {
        (
            self.x0 + col as f32 * self.dx,
            self.y0 + row as f32 * self.dy,
        )
    }
}

pub fn escaped(z: &Complex32) -> bool {
    z.re < -2.0 || z.re > 2.0 || z.im < -2.0 || z.im > 2.0
}

#[kernel]
pub fn julia(
    mut dst: DisjointSlice<u32>,
    ncols: usize,
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
    cx: f32,
    cy: f32,
) {
    let idx = thread::index_1d();
    const COUNT: u32 = 256;

    let grid = JuliaGrid {
        ncols,
        nrows: 0,
        dx,
        dy,
        x0,
        y0,
    };

    if let Some(rval) = dst.get_mut(idx) {
        let col = idx.get() % ncols;
        let row = idx.get() / ncols;
        let (x, y) = grid.xy_for(col, row);

        let c = Complex32::new(cx, cy);
        let z = Complex32::new(x, y);
        *rval = count_julia(z, c, COUNT);
    }
}

pub fn count_julia(z: Complex32, c: Complex32, COUNT: u32) -> u32 {
    let Complex32 { re: x, im: y } = z;
    if false {
        return (y * 255.0) as _;
    } else if true {
        let mut z = Complex32::new(x, y);
        for i in 0..COUNT {
            if escaped(&z) {
                // println!("escaped {z:?} after {i} cycles");
                // return (z.re.abs() * 64.0) as _;
                return i;
            }
            let x1 = z.re * z.re - z.im * z.im + c.re;
            let y1 = z.re * z.im + z.im * z.re + c.im;
            z = Complex::new(x1, y1);
        }
    } else {
        let c = x;
        let mut z = c;
        for i in 0..COUNT {
            if false {
                return i;
            }
            z = z * z + c;
        }
    }

    COUNT
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let grid = {
        let r = 512;
        let ncols = r;
        let nrows = r;
        let width = 4.0;
        let height = 4.0;
        JuliaGrid {
            ncols,
            nrows,
            dx: width / ncols as f32,
            dy: height / nrows as f32,
            x0: -width / 2.0,
            y0: -height / 2.0,
        }
    };
    let count: usize = grid.cell_count();

    println!("doot 1");

    let mut c_dev = DeviceBuffer::<u32>::zeroed(&stream, count).unwrap();

    // Loads `julia-rust-cuda.ptx` directly when cuda-oxide produced PTX, or builds a
    // cubin from `julia-rust-cuda.ll` when cuda-oxide auto-detected libdevice math
    // (`sin`, `pow`, `exp`, ...). Requires CUDA Toolkit on the host.
    let module = load_kernel_module(&ctx, "julia_rust_cuda").expect("Failed to load kernel module");

    println!("doot 2");

    let cx = -0.5125;
    let cy = 0.5213;
    let c_host = if true {
          cuda_launch! {
                kernel: julia,
                stream: stream,
                module: module,
                config: LaunchConfig::for_num_elems(count as u32),
                args: [ slice_mut(c_dev),
                    grid.ncols,
                    grid.x0,
                    grid.y0, grid.dx, grid.dy,
                    cx,
        cy,
                ]
            }
        .expect("Kernel launch failed");

        println!("doot 3");

        c_dev.to_host_vec(&stream).unwrap()
    } else {
        on_cpu(&grid, cx, cy)
    };

    if true {
        println!("{c_host:?}");
    }
    {
        let ofname = "/tmp/x.pgm";
        write_to_pgm(&c_host, grid.ncols, grid.nrows, ofname).unwrap();
        println!("wrote to {ofname}");
    }
}

fn on_cpu(grid: &JuliaGrid, cx: f32, cy: f32) -> Vec<u32> {
    let c = Complex32::new(cx, cy);
    (0..grid.nrows)
        .flat_map(|row| {
            (0..grid.ncols).map(move |col| {
                let (x, y) = grid.xy_for(col, row);
                count_julia(Complex::new(x, y), c, 255)
            })
        })
        .collect()
}

fn write_to_pgm(
    greys: &[u32],
    width: usize,
    height: usize,
    ofname: &str,
) -> Result<(), std::io::Error> {
    let mut f = File::create(ofname)?;
    writeln!(&mut f, "P5\n{} {}\n255\n", width, height)?;
    for g in greys {
        let g8 = [(*g).clamp(0, 255) as u8];
        f.write(&g8)?;
    }
    Ok(())
}
