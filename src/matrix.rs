use std::f64::consts::E;

pub struct Matrix {
  rows: usize,
  cols: usize,
  data: Vec<f32>,
}

impl Matrix {
  
  pub fn new (
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    ) -> Self {
      assert_eq!(rows * cols, data.len(), "Data does not match the matrix dimensions");
      Matrix { rows, cols, data }
  }
  pub fn zeros(
    rows: usize, 
    cols: usize,
    ) -> Self {
      Matrix {
        rows,
        cols,
        data: vec![0.0; rows * cols],
      }
  }
  pub fn identity(
    size: usize,
    ) -> Self {
      let mut data = vec![0.0; size * size];
      for i in 0..size {
        data[i * size + i] = 1.0;
      }
      Matrix {
        rows: size,
        cols: size,
        data, 
      }
  }
  pub fn add(
    &self,
    other: &Matrix,
  ) -> Result<Matrix, &'static str> {
    if self.rows != other.rows || self.cols != other.cols {
      return Err("Matrix dimensions must match for addition");
    }
    let data: Vec<f32> = self
      .data
      .iter()
      .zip(&other.data)
      .map(|(a, b)| a + b)
      .collect();
    Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn elem_add(
    &self, 
    other: &Matrix,
  ) -> Result<Matrix, &'static str> {
    if self.rows != other.rows || self.cols != other.cols {
      return Err("Matrix dimensions must match for element-wise addition");
    }
    let data: Vec<f32> = self
      .data
      .iter()
      .zip(&other.data)
      .map(|(a, b)| a + b)
      .collect();
    Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn add_row_vector(&self, vector: &Matrix) -> Result<Matrix, &'static str> {
      if vector.rows != 1 || vector.cols != self.cols {
          return Err("The vector must have exactly one row and the same number of columns as the matrix");
      }

      let mut data = Vec::with_capacity(self.data.len());
      for i in 0..self.rows {
          for j in 0..self.cols {
              data.push(self.data[i * self.cols + j] + vector.data[j]);
          }
      }

      Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn add_column_vector(&self, vector: &Matrix) -> Result<Matrix, &'static str> {
      if vector.cols != 1 || vector.rows != self.rows {
          return Err("The vector must have exactly one column and the same number of rows as the matrix");
      }

      let mut data = Vec::with_capacity(self.data.len());
      for i in 0..self.rows {
          for j in 0..self.cols {
              data.push(self.data[i * self.cols + j] + vector.data[i]);
          }
      }

      Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn sub(
    &self,
    other: &Matrix,
  ) -> Result<Matrix, &'static str> {
    if self.rows != other.rows || self.cols != other.cols {
      return Err("Matrix dimensions must match for subtraction");
    }
    let data: Vec<f32> = self
      .data
      .iter()
      .zip(&other.data)
      .map(|(a, b)| a - b)
      .collect();
    Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn elem_sub(
    &self,
    other: &Matrix,
  ) -> Result<Matrix, &'static str> {
    if self.rows != other.rows || self.cols != other.cols {
      return Err("Matrix dimensions must match for element-wise subtraction");
    }
    let data: Vec<f32> = self
      .data
      .iter()
      .zip(&other.data)
      .map(|(a, b)| a - b)
      .collect();
    Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn mul(
    &self,
    other: &Matrix,
  ) -> Result<Matrix, &'static str> {
    if self.cols != other.rows {
      return Err("Matrix dimensions must match for multiplication");
    }
    let mut data = vec![0.0; self.rows * other.cols];
    for i in 0..self.rows {
      for j in 0..other.cols {
        data[i * other.cols + j] = (0..self.cols)
          .map(|k| self.data[i * self.cols + k] * other.data[k * other.cols + j])
          .sum();
      }
    }
    Ok(Matrix::new(self.rows, other.cols, data))
  }
  pub fn elem_mul(
    &self,
    other: &Matrix,
  ) -> Result<Matrix, &'static str> {
    if self.rows != other.rows || self.cols != other.cols {
      return Err("Matrix dimensions must match for element-wise multiplication product");
    }
    let data: Vec<f32> = self
      .data
      .iter()
      .zip(&other.data)
      .map(|(a, b)| a * b)
      .collect();
    Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn elem_div(
    &self,
    other: &Matrix,
  ) -> Result<Matrix, &'static str> {
    if self.rows != other.rows || self.cols != other.cols {
      return Err("Matrix dimensions must match for element-wise division");
    }
    let data: Vec<f32> = self
      .data 
      .iter()
      .zip(&other.data)
      .map(|(a, b)| a / b)
      .collect(); 
    Ok(Matrix::new(self.rows, self.cols, data))
  }
  pub fn dot(
    &self, 
    other: &Matrix,
  ) -> Result<f32, &'static str>{
    if self.rows != 1 && self.cols != 1 && other.rows != 1 && other.cols != 1 {
      return Err("Both matrices must be 1D vectors for the dot product");
    }
    if self.rows * self.cols != other.rows * other.cols {
      return Err("Vectors must have the same length for the dot product");
    }
    Ok(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
  }
  pub fn trans(&self) -> Matrix {
    let mut data = vec![0.0; self.rows * self.cols];
    for i in 0..self.rows {
      for j in 0..self.cols {
        data[j * self.rows + i] = self.data[i * self.cols + j];
      }
    }
    Matrix::new(self.cols, self.rows, data)
  }
  pub fn relu(
    &self,
  ) -> Matrix {
    let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
    Matrix::new(self.rows, self.cols, data)
  } 
  pub fn sigmoid(
    &self,
  ) -> Matrix {
    let data: Vec<f32> = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    Matrix::new(self.rows, self.cols, data)
  }
  pub fn tanh(
    &self,
  ) -> Matrix {
    let data: Vec<f32> = self.data.iter().map(|&x| x.tanh()).collect();
    Matrix::new(self.rows, self.cols, data)
  }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_addition() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.add(&m2).unwrap();
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }
    #[test]
    fn test_element_wise_addition() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.elem_add(&m2).unwrap();
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }
    #[test]
    fn test_matrix_subtraction() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.sub(&m2).unwrap();
        assert_eq!(result.data, vec![-4.0, -4.0, -4.0, -4.0]);
    }
    #[test]
    fn test_element_wise_subtraction() {
        let m1 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let m2 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = m1.elem_sub(&m2).unwrap();
        assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);
    }
    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.mul(&m2).unwrap();
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
    #[test]
    fn test_element_wise_multiplication() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.elem_mul(&m2).unwrap();
        assert_eq!(result.data, vec![5.0, 12.0, 21.0, 32.0]);
    }
    #[test]
    fn test_element_wise_division() {
        let m1 = Matrix::new(2, 2, vec![4.0, 9.0, 16.0, 25.0]);
        let m2 = Matrix::new(2, 2, vec![2.0, 3.0, 4.0, 5.0]);
        let result = m1.elem_div(&m2).unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_dot_product() {
        let v1 = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let v2 = Matrix::new(1, 3, vec![4.0, 5.0, 6.0]);
        let result = v1.dot(&v2).unwrap();
        assert_eq!(result, 32.0);
    }
    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = m.trans();
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
    #[test]
    fn test_relu() {
        let m = Matrix::new(2, 2, vec![-1.0, -2.0, 3.0, 4.0]);
        let result = m.relu();
        assert_eq!(result.data, vec![0.0, 0.0, 3.0, 4.0]);
    }
    #[test]
    fn test_sigmoid() {
        let m = Matrix::new(2, 2, vec![0.0, 1.0, 2.0, -1.0]);
        let result = m.sigmoid();
        assert_eq!(
            result.data,
            vec![
                1.0 / (1.0 + E.powf(0.0) as f32),
                1.0 / (1.0 + E.powf(-1.0) as f32),
                1.0 / (1.0 + E.powf(-2.0) as f32),
                1.0 / (1.0 + E.powf(1.0) as f32)
            ]
        );
    }
    #[test]
    fn test_tanh() {
        let m = Matrix::new(2, 2, vec![0.0, 1.0, 2.0, -1.0]);
        let result = m.tanh();
        assert_eq!(
            result.data,
            vec![(0.0 as f32).tanh(), (1.0 as f32).tanh(), (2.0 as f32).tanh(), (-1.0 as f32).tanh()]
        );
    }
    #[test]
    fn test_add_row_vector() {
        let m = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let vector = Matrix::new(1, 3, vec![1.0, 1.0, 1.0]);
        let result = m.add_row_vector(&vector).unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_add_column_vector() {
        let m = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let vector = Matrix::new(3, 1, vec![1.0, 1.0, 1.0]);
        let result = m.add_column_vector(&vector).unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }
}