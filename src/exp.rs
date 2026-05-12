use num_complex::Complex32;

//

pub trait ComplexNumber: for<'a> std::ops::Add<&'a Self, Output = Self> + Sized
where
    for<'a> &'a Self: std::ops::Mul<&'a Self, Output = Self>,
{
    fn build(re: f32, im: f32) -> Self;

    fn re(&self) -> f32;
    fn im(&self) -> f32;
}

//

#[derive(Clone, Copy)]
pub struct MyComplex {
    pub re: f32,
    pub im: f32,
}

impl MyComplex {
    pub fn new(re: f32, im: f32) -> MyComplex {
        Self { re, im }
    }
}

impl std::ops::Mul for MyComplex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self) * &rhs
    }
}

impl std::ops::Mul for &MyComplex {
    type Output = MyComplex;

    fn mul(self, rhs: Self) -> Self::Output {
        MyComplex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Add<&Self> for MyComplex {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl From<&Complex32> for MyComplex {
    fn from(c: &Complex32) -> Self {
        Self { re: c.re, im: c.im }
    }
}

impl Into<Complex32> for MyComplex {
    fn into(self) -> Complex32 {
        Complex32 {
            re: self.re,
            im: self.im,
        }
    }
}

impl ComplexNumber for MyComplex {
    fn build(re: f32, im: f32) -> MyComplex {
        Self { re, im }
    }

    fn re(&self) -> f32 {
        self.re
    }

    fn im(&self) -> f32 {
        self.im
    }
}

//

impl ComplexNumber for Complex32 {
    fn build(re: f32, im: f32) -> Complex32 {
        Complex32 { re, im }
    }

    fn re(&self) -> f32 {
        self.re
    }

    fn im(&self) -> f32 {
        self.im
    }
}
