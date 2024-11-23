use bulletformat::{chess::BoardIter, ChessBoard};

use super::InputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Chess192;
impl InputType for Chess192 {
    type RequiredDataType = ChessBoard;
    type FeatureIter = Chess192Iter;

    fn max_active_inputs(&self) -> usize {
        32
    }

    fn inputs(&self) -> usize {
        192
    }

    fn buckets(&self) -> usize {
        1
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        Chess192Iter { board_iter: pos.into_iter() }
    }
}

pub struct Chess192Iter {
    board_iter: BoardIter,
}

impl Iterator for Chess192Iter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(piece, square)| {
            const HALF_INPUT_SIZE: usize = 192 / 2;
            const SQUARE_SIZE: usize = 8 + 8; 

            let c = usize::from(piece & 8 > 0);
            let pc = SQUARE_SIZE * usize::from(piece & 7);
            let sq = usize::from(square);
            let sq_flipped = sq ^ 56;

            let wsq = 8 * (sq % 8) + (sq / 8);
            let bsq = 8 * (sq_flipped % 8) + (sq_flipped / 8);

            let wfeat = [0, HALF_INPUT_SIZE][c] + pc + wsq;
            let bfeat = [HALF_INPUT_SIZE, 0][c] + pc + bsq;
            (wfeat, bfeat)
        })
    }
}
