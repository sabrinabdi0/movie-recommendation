#!/usr/bin/env python3
"""Download MovieLens data and build the content-based recommender model."""

from recommender import MovieRecommender


def main() -> None:
    recommender = MovieRecommender()
    recommender.train()


if __name__ == "__main__":
    main()
