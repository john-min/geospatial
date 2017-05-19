#csvkit shell script

csvsql --query "select * from top_active_devices where rank<=10" top_active_devices.csv > top10_devices.csv

if [ 1 -eq 0 ]; then

  csvsql --query "select * from ip_active_device where udid in (
    '7022a104-c709-4e82-b88e-3318c529e32e',
    '657ae5e1-a094-45d0-8731-f9af8af72493',
    '6c59c6db-34fa-4156-adec-8cef7811048a',
    'bfe92148-875b-4e49-b497-9ee3304dc6c8',
    'd4636720-833a-43f6-90f2-061bbd6fd070',
    'DC1A085E-FFB4-4528-A318-1BE453BDCE79',
    '73cb6a78-cf9c-4f07-aaeb-a964fbc457c9',
    '92B20754-C58A-4C2B-A0F7-F5236C3F6F36',
    'a84519ca-1c65-468a-bcf2-2b1b7af3544b',
    '49fe3d7e-328d-4402-ab00-d93e40523308')" ip_active_device.csv > top10_devices_data.csv

  echo "The code that you want commented out goes here."
  echo "This echo statement will not be called."
fi

csvjoin ip_active_device.csv top_active_devices.csv > joined.csv
#csvjoin -c "udid, rank" ip_active_device.csv top_active_devices.csv > joined.csv
#csvjoin -c udid ip_active_device.csv top_active_devices.csv > joined.csv
csvsql --query "select * from joined where rank<=10" joined.csv > top10_devices_data.csv
rm joined.csv
