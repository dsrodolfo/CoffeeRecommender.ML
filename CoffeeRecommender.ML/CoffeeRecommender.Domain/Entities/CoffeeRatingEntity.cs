using Microsoft.ML.Data;

namespace CoffeeRecommender.Domain.Entities
{
    class CoffeeRatingEntity
    {
        [LoadColumn(0)]
        public float UserId { get; set; }
        [LoadColumn(1)]
        public float CoffeeId { get; set; }
        [LoadColumn(2)]
        public float Rating { get; set; }
    }
}