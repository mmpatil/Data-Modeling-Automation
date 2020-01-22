'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('ShortlistedModels', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      RunId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'RunDetail',
          key: 'Id'
        }
      },
      ModelId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'ModelRunDetail',
          key: 'Id'
        }
      }
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('ShortlistedModels');
  }
};
