use crate::exp::MyComplex;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::cuda_module;
use std::fs::File;
use std::io::Write;
// type MyComplex = num_complex::Complex32;

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

#[cuda_module]
mod kernels {
    use crate::JuliaGrid;
    use crate::MyComplex;
    use cuda_device::{kernel, thread, DisjointSlice};

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn julia(
        mut dst: DisjointSlice<usize>,
        ncols: usize,
        x0: f32,
        y0: f32,
        dx: f32,
        dy: f32,
        cx: f32,
        cy: f32,
        max_iter: usize,
    ) {
        let idx = thread::index_1d();

        let grid = JuliaGrid {
            ncols,
            nrows: 0,
            dx,
            dy,
            x0,
            y0,
        };

        // I wish get_mut did not consume idx
        let col = idx.get() % ncols;
        let row = idx.get() / ncols;
        if let Some(rval) = dst.get_mut(idx) {
            let (x, y) = grid.xy_for(col, row);

            let c = MyComplex::new(cx, cy);
            let z = MyComplex::new(x, y);
            *rval = super::count_julia(z, c, max_iter);
        }
    }
}

pub fn count_julia(mut z: MyComplex, c: MyComplex, max_iter: usize) -> usize {
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

fn main() {
    match 2 {
        2 => animation1::animation1(),
        _ => still1(),
    };
}

fn still1() {
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
        on_gpu(&grid, cx, cy, 256)
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

mod animation1 {
    use crate::{colormap, on_gpu, JuliaGrid};
    use std::fs::File;

    pub fn animation1() {
        let mut radius = 2.0;

        let julia_c = (-0.5125, 0.5213);
        let center = (0.0, 0.0);

        for i in 0..10 {
            compute_one_frame(i, julia_c, center, radius);
            radius *= 0.5;
        }
    }

    fn compute_one_frame(i: i32, julia_c: (f32, f32), center: (f32, f32), radius: f32) {
        let width = 512;
        let delta = radius * 2.0 / width as f32;
        let grid = JuliaGrid {
            ncols: width,
            nrows: width,
            dx: delta,
            dy: delta,
            x0: center.0 - radius,
            y0: center.1 - radius,
        };
        let counts = on_gpu(&grid, julia_c.0, julia_c.1, 1024);

        let rgbs = colormap(&counts);

        let ofname = format!("/tmp/julia-1/image{i}.png");

        let f = File::create(ofname).unwrap();
        let mut encoder = png::Encoder::new(f, width as u32, width as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&rgbs).unwrap();
    }
}

pub fn colormap(counts: &[usize]) -> Vec<u8> {
    counts.iter().copied().flat_map(color_for).collect()
}

pub fn color_for(count: usize) -> [u8; 3] {
    const COLORS: [[u8; 3]; 6] = [
        [255, 0, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [255, 0, 255],
    ];

    const Q: usize = 20;
    let phase = count % (COLORS.len() * Q);
    let block = phase / Q;
    let small = phase % Q;
    let c1 = &COLORS[block];
    let c2 = &COLORS[(block + 1) % COLORS.len()];
    [
        ((c2[0] as usize * small + c1[0] as usize * (Q - small)) / Q) as u8,
        ((c2[1] as usize * small + c1[1] as usize * (Q - small)) / Q) as u8,
        ((c2[2] as usize * small + c1[2] as usize * (Q - small)) / Q) as u8,
    ]
}

fn on_gpu(grid: &JuliaGrid, cx: f32, cy: f32, max_iter: usize) -> Vec<usize> {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    println!("doot 1");

    let count: usize = grid.cell_count();

    let mut c_dev = DeviceBuffer::<usize>::zeroed(&stream, count).unwrap();

    // Loads `julia-rust-cuda.ptx` directly when cuda-oxide produced PTX, or builds a
    // cubin from `julia-rust-cuda.ll` when cuda-oxide auto-detected libdevice math
    // (`sin`, `pow`, `exp`, ...). Requires CUDA Toolkit on the host.
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    println!("doot 2");

    module
        .julia(
            &stream,
            LaunchConfig::for_num_elems(count as u32),
            &mut c_dev,
            grid.ncols,
            grid.x0,
            grid.y0,
            grid.dx,
            grid.dy,
            cx,
            cy,
            max_iter,
        )
        .expect("Kernel launch failed");

    println!("doot 3");

    c_dev.to_host_vec(&stream).unwrap()
}

fn on_cpu(grid: &JuliaGrid, cx: f32, cy: f32) -> Vec<usize> {
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
    greys: &[usize],
    width: usize,
    height: usize,
    ofname: &str,
) -> Result<(), std::io::Error> {
    let mut f = File::create(ofname)?;
    writeln!(&mut f, "P5\n{} {}\n255\n", width, height)?;
    for g in greys {
        let g8 = [(*g).clamp(0, 255) as u8];
        f.write_all(&g8)?;
    }
    Ok(())
}
