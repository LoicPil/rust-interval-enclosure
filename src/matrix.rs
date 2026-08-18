use inari::Interval;

fn matmul_naive(a: &[Interval], b: &[Interval], m: usize, k: usize, n: usize) -> Vec<Interval> {
    let mut c = vec![Interval::EMPTY; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Interval::try_from((0.0, 0.0)).unwrap();
            for p in 0..k {
                acc = acc + a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::matmul_naive;
    use inari::{Interval, interval}; // adapte le chemin si ta fonction est ailleurs

    #[test]
    fn test_a_times_a() {
        // A = | 1        [0,1] |
        //     | 1        -1    |
        let one = Interval::try_from((1.0, 1.0)).unwrap();
        let neg_one = Interval::try_from((-1.0, -1.0)).unwrap();
        let zero_one = interval!(0.0, 1.0).unwrap();

        let a: Vec<Interval> = vec![one, zero_one, one, neg_one];

        let c = matmul_naive(&a, &a, 2, 2, 2);

        // Expected:
        // [1,2]   [-1,1]
        // [0,0]   [1,2]
        assert_eq!(c[0], interval!(1.0, 2.0).unwrap());
        assert_eq!(c[1], interval!(-1.0, 1.0).unwrap());
        assert_eq!(c[2], interval!(0.0, 0.0).unwrap());
        assert_eq!(c[3], interval!(1.0, 2.0).unwrap());
    }
}
