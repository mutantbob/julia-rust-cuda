use crate::exp::MyComplex;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::{cuda_launch, load_kernel_module};
use num_complex::{Complex, Complex32};
use std::fs::File;
use std::io::Write;

// type MyComplex = Complex32;

pub mod exp;

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

pub fn escaped(z: &MyComplex) -> bool {
    z.re < -2.0 || z.re > 2.0 || z.im < -2.0 || z.im > 2.0
}

#[kernel]
pub fn julia(mut dst: DisjointSlice<u32>, grid: &JuliaGrid, cx: f32, cy: f32) {
    let idx = thread::index_1d();
    const COUNT: u32 = 256;

    if let Some(rval) = dst.get_mut(idx) {
        let col = idx.get() % grid.ncols;
        let row = idx.get() / grid.ncols;
        let (x, y) = grid.xy_for(col, row);

        let c = MyComplex::new(cx, cy);
        let z = MyComplex::new(x, y);
        *rval = count_julia(z, c, COUNT);
    }
}

pub fn count_julia(mut z: MyComplex, c: MyComplex, max_iter: u32) -> u32 {
    for i in 0..max_iter {
        if escaped(&z) {
            // println!("escaped {z:?} after {i} cycles");
            // return (z.re.abs() * 64.0) as _;
            return i;
        }
        z = julia_one_iter(&c, &z);

        // let tmp = std::ops::Mul::mul(z, z);
        // z = z*z+c;
    }

    max_iter
}

fn julia_one_iter(c: &MyComplex, z: &MyComplex) -> MyComplex {
    z * z + c
}

fn julia_one_iter_orig(c: &Complex32, z: &Complex32) -> Complex32 {
    let x1 = z.re * z.re - z.im * z.im + c.re;
    let y1 = z.re * z.im + z.im * z.re + c.im;
    Complex::new(x1, y1)
}

fn main() {
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
    let cx = -0.5125;
    let cy = 0.5213;
    let c_host = if true {
        on_gpu(&grid, cx, cy)
    } else {
        on_cpu(&grid, cx, cy)
    };

    if false {
        println!("{c_host:?}");
    }
    {
        let ofname = "/tmp/x.pgm";
        write_to_pgm(&c_host, grid.ncols, grid.nrows, ofname).unwrap();
        println!("wrote to {ofname}");
    }
}

fn on_gpu(grid: &JuliaGrid, cx: f32, cy: f32) -> Vec<u32> {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    println!("doot 1");

    let count: usize = grid.cell_count();

    let mut c_dev = DeviceBuffer::<u32>::zeroed(&stream, count).unwrap();

    // Loads `julia-rust-cuda.ptx` directly when cuda-oxide produced PTX, or builds a
    // cubin from `julia-rust-cuda.ll` when cuda-oxide auto-detected libdevice math
    // (`sin`, `pow`, `exp`, ...). Requires CUDA Toolkit on the host.
    let module = load_kernel_module(&ctx, "julia_rust_cuda").expect("Failed to load kernel module");

    println!("doot 2");

    cuda_launch! {
        kernel: julia,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(count as u32),
        args: [ slice_mut(c_dev),
            &grid,
            cx,cy,
        ]
    }
    .expect("Kernel launch failed");

    println!("doot 3");

    c_dev.to_host_vec(&stream).unwrap()
}

fn on_cpu(grid: &JuliaGrid, cx: f32, cy: f32) -> Vec<u32> {
    let c = MyComplex::new(cx, cy);
    (0..grid.nrows)
        .flat_map(|row| {
            (0..grid.ncols).map(move |col| {
                let (x, y) = grid.xy_for(col, row);
                count_julia(MyComplex::new(x, y), c, 255)
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
