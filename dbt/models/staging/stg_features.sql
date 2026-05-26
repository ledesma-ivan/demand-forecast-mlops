with source as (
    select * from {{ source('raw', 'raw_features') }}
),

cleaned as (
    select
        Store                           as store_id,
        cast(Date as date)              as sale_date,
        Temperature                     as temperature,
        Fuel_Price                      as fuel_price,
        coalesce(MarkDown1, 0)          as markdown_1,
        coalesce(MarkDown2, 0)          as markdown_2,
        coalesce(MarkDown3, 0)          as markdown_3,
        coalesce(MarkDown4, 0)          as markdown_4,
        coalesce(MarkDown5, 0)          as markdown_5,
        CPI                             as cpi,
        Unemployment                    as unemployment
    from source
)

select * from cleaned
