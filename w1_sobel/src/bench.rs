extern crate test;
use test::Bencher;

#[bench]
fn bench_xor_1000_ints(b: &mut Bencher) {
    b.iter(|| {
        // Use `test::black_box` to prevent compiler optimizations from disregarding
        // Unused values
        test::black_box(range(0u, 1000).fold(0, |old, new| old ^ new));
    });
}
