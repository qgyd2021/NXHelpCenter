#!/usr/bin/env bash

# 参数:
elasticsearch_version="8.8.2";
system_version="centos";
deploy_mode="docker"

# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done

echo "elasticsearch_version: ${elasticsearch_version}";
echo "system_version: ${system_version}";


if [ ${system_version} = "windows" ]; then
  #https://www.elastic.co/cn/downloads/elasticsearch
  cd '/d/ProgramFiles' || exit 1;

  #wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.8.2-windows-x86_64.zip
  #unzip elasticsearch-8.8.2-windows-x86_64.zip

  wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.12-windows-x86_64.zip
  unzip elasticsearch-7.17.12-windows-x86_64.zip

  #Windows 平台安装后, 双击 bin/elasticsearch.bat 以启动服务.
  #Terminal 执行 curl -X GET "localhost:9200/?pretty" 以检索是否启动成功.

elif [ ${system_version} = "centos" ]; then

  if [ ${deploy_mode} = "docker" ]; then
    #https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
    sysctl -w vm.max_map_count=262144
    sysctl -p

    docker pull docker.elastic.co/elasticsearch/elasticsearch:7.17.12
    docker run -itd -p 127.0.0.1:9200:9200 -p 127.0.0.1:9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.17.12

    # curl -X GET "localhost:9200/?pretty"
  fi

fi
