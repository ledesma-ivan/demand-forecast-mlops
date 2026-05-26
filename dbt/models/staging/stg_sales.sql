with source as (
    select * from {{ source('raw', 'raw_train') }}
),

renamed as (
    select
        Store                            as store_id,
        Dept                             as dept_id,
        cast(Date as date)               as sale_date,
        Weekly_Sales                     as weekly_sales,
        cast(IsHoliday as boolean)       as is_holiday
    from source
)

select * from renamed
